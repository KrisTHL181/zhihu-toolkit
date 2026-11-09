"""
Zhihu Backup to Markdown Converter

Convert Zhihu HTML content (answers, articles, pins) to Markdown format.
Adapted from the Tampermonkey script "zhihu-backup-collect".
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from lxml.html import HtmlElement

_lxml_html = None


def _get_lxml_html() -> Any:
    """Import and cache the ``lxml.html`` module on first use.

    ``lxml`` is a heavy C-extension dependency; importing it eagerly would add
    ~30 ms to every CLI invocation. It is only needed when HTML is actually
    parsed, so defer it until first conversion.

    :returns: The ``lxml.html`` module.
    """
    global _lxml_html
    if _lxml_html is None:
        from lxml import html as _lxml_html
    return _lxml_html


_eeimg_re = re.compile(r"eeimg|equation")

#: Fallback separator between adjacent inline elements whose delimiters merge.
#: CommonMark joins back-to-back delimiter runs, so ``**a****b**`` is parsed as
#: one strong span instead of two. Same-type runs are merged where safe; when
#: merging is ambiguous (mixed or nested delimiters), an empty HTML comment
#: keeps the runs apart without rendering any content.
_INLINE_SEPARATOR = "<!-- -->"

#: Characters that open/close a Markdown inline delimiter (emphasis and code).
_DELIMITER_CHARS: frozenset[str] = frozenset("*_~`")


def _delimiters_collide(previous: str, current: str) -> bool:
    """Return whether adjacent rendered parts would merge into one delimiter run.

    :param previous: The Markdown rendered for the preceding element.
    :param current: The Markdown rendered for the current element.
    :returns: ``True`` when *previous* ends with the delimiter *current* starts with.
    """
    return bool(previous) and bool(current) and previous[-1] == current[0] and previous[-1] in _DELIMITER_CHARS


def _simple_delimited(part: str) -> tuple[str, str] | None:
    """Return ``(delimiter, inner)`` when *part* is a simple delimited run.

    Simple means one delimiter pair wrapping text that contains no ``*`` (for
    emphasis) or backtick (for code). Nested emphasis such as ``***a***`` is
    not simple: merging it with a neighbour would produce an ambiguous
    delimiter run.

    :param part: The Markdown rendered for one inline element.
    :returns: The delimiter and inner text, or ``None`` when not simple.
    """
    for delimiter in ("**", "*"):
        if part.startswith(delimiter) and part.endswith(delimiter) and len(part) > 2 * len(delimiter):
            inner = part[len(delimiter) : -len(delimiter)]
            if inner and "*" not in inner:
                return delimiter, inner
    if part.startswith("`") and part.endswith("`") and len(part) > 2:
        inner = part[1:-1]
        if inner and "`" not in inner:
            return "`", inner
    return None


def _merge_delimited(previous: str, current: str) -> str | None:
    """Merge two simple adjacent inline runs sharing the same delimiter.

    Two sibling elements of the same inline type are a single semantic run in
    HTML, so ``<strong>a</strong><strong>b</strong>`` should become ``**ab**``
    rather than two delimiter runs that need an invisible separator. Merging is
    only safe when both sides are *simple* runs (:func:`_simple_delimited`) with
    the same delimiter; nested emphasis and mixed types fall back to
    :data:`_INLINE_SEPARATOR`.

    :param previous: The Markdown rendered for the preceding element.
    :param current: The Markdown rendered for the current element.
    :returns: The merged run, or ``None`` when no safe merge exists.
    """
    left_run = _simple_delimited(previous)
    right_run = _simple_delimited(current)
    if left_run is None or right_run is None or left_run[0] != right_run[0]:
        return None
    delimiter, left = left_run
    _, right = right_run
    return f"{delimiter}{left}{right}{delimiter}"


def _is_simple_mixed_emphasis(previous: str, current: str) -> bool:
    """Return whether two simple emphasis runs use different delimiters.

    ``<em>a</em><strong>b</strong>`` is unambiguous as ``*a***b**`` and needs
    no separator, unlike a pair of nested runs.

    :param previous: The Markdown rendered for the preceding element.
    :param current: The Markdown rendered for the current element.
    :returns: ``True`` when both are simple emphasis runs of different kinds.
    """
    left_run = _simple_delimited(previous)
    right_run = _simple_delimited(current)
    return bool(
        left_run
        and right_run
        and left_run[0] in ("**", "*")
        and right_run[0] in ("**", "*")
        and left_run[0] != right_run[0]
    )


# ── lxml helpers ──────────────────────────────────────────────────────────────


def replace_with_text(elem: HtmlElement, text: str) -> None:
    """Replace an lxml element with a text node, preserving surrounding tail text.

    lxml has no direct equivalent of BeautifulSoup's ``replace_with()``.
    This inserts *text* at the element's position and appends the element's
    original ``.tail`` so text that follows the replaced element is kept.
    """
    parent = elem.getparent()
    if parent is None:
        return
    tail = elem.tail or ""
    prev = elem.getprevious()
    if prev is not None:
        prev.tail = (prev.tail or "") + text + tail
    else:
        parent.text = (parent.text or "") + text + tail
    parent.remove(elem)


def _iter_nodes(element: HtmlElement):
    """Yield text strings and child elements in document order.

    lxml's tree model is different from BeautifulSoup's:

    * ``element.text``  – text before the first child
    * ``child.tail``    – text after each child element

    BeautifulSoup treats text nodes as children (NavigableString);
    this generator bridges the gap by yielding ``str`` for text and
    ``HtmlElement`` for tags, matching the BS ``.children`` contract.
    """
    if element.text:
        yield element.text
    for child in element:
        yield child
        if child.tail:
            yield child.tail


def _tag_name(element: HtmlElement) -> str:
    """Return the lowercase tag name of an lxml element."""
    tag = element.tag
    return tag.lower() if isinstance(tag, str) else str(tag)


def image_src(element: HtmlElement) -> str:
    """Return the real image URL carried by an ``<img>`` element.

    Zhihu lazy-loads images: ``src`` often holds a placeholder while the
    actual picture sits in a ``data-*`` attribute. Prefer the first absolute
    candidate, since trusting ``src`` alone yields the placeholder.

    Falls back to ``src`` unchanged when no absolute candidate exists, so
    inline ``data:`` URIs and relative URLs keep their previous behaviour.

    :param element: An lxml ``<img>`` element.
    :returns: The image URL, or ``""`` when the element carries none.
    """
    for attr in ("data-original", "data-actualsrc", "data-src", "src"):
        value = (element.get(attr) or "").strip()
        if value.startswith("http"):
            return value
    return (element.get("src") or "").strip()


# ── link converter ────────────────────────────────────────────────────────────


class ZhihuLinkConverter:
    """Convert Zhihu internal links to normal URLs."""

    @staticmethod
    def normalize_link(link: str) -> str:
        """Convert a Zhihu link to its original target."""
        if not link:
            return link

        # Handle link.zhihu.com redirections
        if "link.zhihu.com" in link:
            parsed = urlparse(link)
            query = parse_qs(parsed.query)
            target = query.get("target", [None])[0]
            if target:
                return target
        return link


# ── public API ────────────────────────────────────────────────────────────────


class PageToMarkdown:
    """
    Convert Zhihu page HTML content to Markdown.
    """

    def __init__(self, skip_empty: bool = True) -> None:
        self.skip_empty = skip_empty
        self.link_converter = ZhihuLinkConverter()
        # The module-level ``converter`` singleton is shared by parallel
        # download workers, and ``convert()`` resets the citation collection on
        # every call. A plain attribute would let one thread wipe another's
        # citations mid-conversion, so the collection is stored per thread.
        self._local = threading.local()

    @property
    def _references(self) -> dict[str, tuple[str, str]]:
        """Citations collected by this thread during the current conversion.

        :returns: The per-thread ``number -> (title, url)`` mapping, created
            lazily on first access.
        """
        references = getattr(self._local, "references", None)
        if references is None:
            references = {}
            self._local.references = references
        return references

    @_references.setter
    def _references(self, value: dict[str, tuple[str, str]]) -> None:
        self._local.references = value

    # ── LaTeX pre-processing ──────────────────────────────────────────────

    def tex_normalize(self, content: str) -> str:
        """
        Convert inline-mode
         <img eeimg="1" src="https://www.zhihu.com/equation?tex=A" alt="B"/>
        To:
         $B$

        Convert display-mode
         ![A](https://www.zhihu.com/equation?tex=B)
        to
         $$B$$
        """
        if not content:
            return ""

        pattern = r"!\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(https://www\.zhihu\.com/equation\?tex=[^)]*\)"

        content = re.sub(pattern, lambda match: f"$${match.group(1)}$$", content)

        doc = _get_lxml_html().fromstring(content)

        # self:: axis covers the root element when content is a single img fragment
        for img in doc.xpath(".//img[@eeimg='1'] | self::img[@eeimg='1']"):
            latex_content = img.get("alt", "")
            if latex_content:
                if img is doc:
                    return f"${latex_content}$"
                replace_with_text(img, f"${latex_content}$")
        # Handle block/display mode formulas (eeimg="2")
        for img in doc.xpath(".//img[@eeimg='2'] | self::img[@eeimg='2']"):
            latex_content = img.get("alt", "")
            if latex_content:
                if img is doc:
                    return f"\n$$\n{latex_content}\n$$\n"
                replace_with_text(img, f"\n$$\n{latex_content}\n$$\n")
        return _get_lxml_html().tostring(doc, encoding="unicode")

    # ── top-level entry points ───────────────────────────────────────────

    def convert(self, html_content: str, url: str = "", strip: bool = True) -> str:
        """
        Convert HTML content to Markdown.

        This is the public entry point: it normalizes LaTeX images first, then
        walks the document, then strips the result.

        Inline Zhihu citations become ``[^n]`` footnotes and their targets are
        collected into a trailing ``参考`` section.

        :param html_content: HTML string of the Zhihu page.
        :param url: Base URL for resolving relative links (optional).
        :param strip: Strip leading/trailing whitespace from the result.
        :return: Markdown string.
        """
        if not html_content:
            return ""

        self._references = {}
        content = self.tex_normalize(html_content)
        result = self._convert(content, url)
        result = self._append_references(result)
        return result.strip() if strip else result

    def _convert(self, html_content: str, url: str = "") -> str:
        """
        Walk already-normalized HTML and return Markdown.

        Call :meth:`convert` instead unless the content has been through
        :meth:`tex_normalize` already.

        :param html_content: The HTML string to convert.
        :param url: Optional URL for resolving relative links.
        :return: Markdown string.
        """
        if not html_content:
            return ""

        doc = _get_lxml_html().fromstring(html_content)

        # Remove useless elements
        for element in doc.xpath(".//script") + doc.xpath(".//style"):
            element.drop_tree()

        # A page scraped as HTML carries the citation list Zhihu rendered server
        # side, which would duplicate the one rebuilt from the inline markers.
        if doc.cssselect('sup[data-draft-type="reference"]'):
            self._drop_rendered_references(doc)

        # Handle Zhihu full pages (with .RichText container)
        rich_texts = doc.cssselect(".RichText")
        if rich_texts:
            markdown_parts = []
            for rt in rich_texts:
                md = self._dispatch(rt)
                if md:
                    markdown_parts.append(md)
            return "\n\n".join(markdown_parts)

        # Full HTML document – iterate children of <html>
        if doc.tag == "html":
            parts = []
            for node in _iter_nodes(doc):
                md = self._dispatch(node)
                if md:
                    parts.append(md)
            return "\n\n".join(parts) if parts else ""

        # HTML fragment – process the root element directly
        md = self._dispatch(doc)
        return md if md else ""

    @staticmethod
    def _drop_rendered_references(doc: HtmlElement) -> None:
        """Remove a server-rendered ``参考`` list from *doc*.

        Only called when the document also carries inline citation markers, so
        the list is never the sole record of the references.

        :param doc: The parsed document to prune in place.
        """
        for ol in doc.cssselect("ol.ReferenceList"):
            previous = ol.getprevious()
            if (
                previous is not None
                and _tag_name(previous) == "h2"
                and (previous.text_content() or "").strip() == "参考"
            ):
                previous.drop_tree()
            ol.drop_tree()

    def _append_references(self, markdown: str) -> str:
        """Append the footnote definitions collected during conversion.

        Zhihu ships the ``参考`` list only in the server-rendered page, which the
        scrapers never read; the URLs ride along on the inline citation markers,
        so the list is rebuilt here from what :meth:`_render_reference` saw.
        A citation title, when the author gave one, precedes its URL.

        :param markdown: The converted document body.
        :returns: *markdown* followed by a ``参考`` section, or *markdown*
            unchanged when nothing was cited.
        """
        if not self._references:
            return markdown

        def sort_key(item: tuple[str, tuple[str, str]]) -> tuple[int, int | str]:
            number = item[0]
            return (0, int(number)) if number.isdigit() else (1, number)

        def definition(number: str, reference: tuple[str, str]) -> str:
            title, target = reference
            label = f"{title} — " if title and title != target else ""
            return f"[^{number}]: {label}{target}"

        definitions = "\n".join(
            definition(number, reference) for number, reference in sorted(self._references.items(), key=sort_key)
        )
        section = f"## 参考\n\n{definitions}"
        body = markdown.strip()
        return f"{body}\n\n{section}" if body else section

    # ── dispatcher ───────────────────────────────────────────────────────

    def _dispatch(self, element) -> str | None:
        """Recursively convert a single element to Markdown.

        Accepts either an lxml ``HtmlElement`` or a plain ``str`` (text node).
        Constructs identified by tag alone go through :data:`_TAG_HANDLERS`;
        the attribute-conditional ones are guarded here, because each must win
        over the generic handler for its own tag.

        :param element: An lxml element or a plain text node.
        :returns: The Markdown for *element*, or ``None`` when it contributes nothing.
        """
        if isinstance(element, str):
            return element.strip() or None

        # Safety net – not an element
        if not hasattr(element, "tag"):
            return None

        tag = _tag_name(element)
        if tag in _SKIP_TAGS:
            return None

        # Order is semantic: every one of these must be tested before the
        # generic handler for the same tag.
        if tag == "div" and _is_link_card(element):
            return self._render_link_card(element)
        if tag == "a" and _is_discarded_link(element):
            return None
        if tag == "span" and _is_math_span(element):
            return self._render_math(element)
        if tag == "div" and _is_video_div(element):
            return self._render_video(element)
        if tag == "sup" and _is_reference_sup(element):
            return self._render_reference(element)

        handler = _TAG_HANDLERS.get(tag)
        if handler is None:
            return self._render_children(element)
        return handler(self, element)

    # ── inline descent ───────────────────────────────────────────────────

    def _render_inline(self, element) -> str:
        """Process inline content, preserving the source's whitespace.

        Adjacent inline elements are concatenated without inserted spaces: the
        HTML already says where spaces belong, and adding one around every
        ``<strong>`` / ``<span>`` / ``<a>`` corrupts CJK text and separates
        punctuation. Text nodes collapse runs of HTML whitespace to a single
        space, keeping the leading/trailing space that separates them from a
        neighbouring element. Two adjacent elements of the same inline type
        are merged where unambiguous (``**a**`` + ``**b**`` -> ``**ab**``),
        as are simple mixed emphasis pairs (``*a*`` + ``**b**`` ->
        ``*a***b**``); any other colliding delimiter run is kept apart by an
        invisible :data:`_INLINE_SEPARATOR` comment.

        :param element: An lxml element or a plain text node.
        :returns: The inline Markdown, or ``""`` when there is none.
        """
        if isinstance(element, str):
            return element.strip()

        if not hasattr(element, "tag"):
            return ""

        parts: list[str] = []
        previous: str | None = None
        previous_is_element = False
        for node in _iter_nodes(element):
            if isinstance(node, str):
                text = re.sub(r"\s+", " ", node)
                if not text:
                    continue
                parts.append(text)
                previous = text
                previous_is_element = False
            else:
                md = self._dispatch(node)
                if not md:
                    continue
                if previous_is_element and previous is not None:
                    merged = _merge_delimited(previous, md)
                    if merged is not None:
                        parts[-1] = merged
                        previous = merged
                        continue
                    if _is_simple_mixed_emphasis(previous, md):
                        combined = previous + md
                        parts[-1] = combined
                        previous = combined
                        continue
                    if _delimiters_collide(previous, md):
                        parts.append(_INLINE_SEPARATOR)
                parts.append(md)
                previous = md
                previous_is_element = True
        return "".join(parts).strip()

    # ── handlers ─────────────────────────────────────────────────────────

    def _render_heading(self, element) -> str:
        """Render ``<h1>``–``<h6>`` as an ATX heading."""
        level = int(_tag_name(element)[1])
        return f"{'#' * level} {self._render_inline(element)}"

    def _render_paragraph(self, element) -> str | None:
        """Render ``<p>``, skipping it entirely when empty and ``skip_empty`` is set."""
        if self.skip_empty and not element.text_content().strip():
            return None
        return self._render_inline(element)

    def _render_unordered_list(self, element) -> str | None:
        """Render ``<ul>`` as a bullet list."""
        items = [f"- {self._render_inline(li)}" for li in element.findall("li")]  # direct children only
        return "\n".join(items) if items else None

    def _render_ordered_list(self, element) -> str | None:
        """Render ``<ol>`` as a numbered list."""
        items = [
            f"{idx}. {self._render_inline(li)}"  # direct children only
            for idx, li in enumerate(element.findall("li"), start=1)
        ]
        return "\n".join(items) if items else None

    def _render_blockquote(self, element) -> str:
        """Render ``<blockquote>`` with every line prefixed by ``> ``."""
        content = self._render_inline(element)
        return "\n".join(f"> {line}" for line in content.split("\n"))

    def _render_code_block(self, element) -> str:
        """Render ``<pre>`` as a fenced code block, honouring a ``language-*`` class."""
        language = ""
        code_tag = element.find(".//code")
        if code_tag is not None:
            for cls in code_tag.classes:
                if cls.startswith("language-"):
                    language = cls.split("-")[1]
                    break
        return f"```{language}\n{element.text_content()}\n```"

    def _render_rule(self, element) -> str:
        """Render ``<hr>`` as a thematic break."""
        return "---"

    def _render_image(self, element) -> str:
        """Render ``<img>`` as a Markdown image."""
        alt = element.get("alt", "")
        src = self.link_converter.normalize_link(image_src(element))
        return f"![{alt}]({src})"

    def _render_figure(self, element) -> str:
        """Render ``<figure>`` as its image plus an optional italic caption.

        :returns: The Markdown, which is empty when the figure holds neither an image nor a caption.
        """
        img = element.find(".//img")
        if img is not None:
            alt = img.get("alt", "")
            src = self.link_converter.normalize_link(image_src(img))
            md_img = f"![{alt}]({src})"
        else:
            md_img = ""

        figcaption = element.find(".//figcaption")
        if figcaption is not None:
            md_img += f"\n*{figcaption.text_content().strip()}*"

        return md_img

    def _render_link(self, element) -> str:
        """Render ``<a>`` as a Markdown link, falling back through its text sources."""
        href = element.get("href", "")
        text = element.text_content().strip()
        if not text:
            text = element.get("title") or element.get("data-text") or href
        return f"[{text}]({self.link_converter.normalize_link(href)})"

    def _render_link_card(self, element) -> str:
        """Render a Zhihu link card (``div.RichText-LinkCardContainer``).

        The card title prefers ``data-text``, unlike a plain ``<a>``.
        """
        a_tag = element.find(".//a")
        if a_tag is None:
            return self._render_inline(element)

        href = a_tag.get("href", "")
        text = a_tag.get("data-text") or a_tag.text_content().strip()
        if not text:
            text = href
        return f"[{text}]({self.link_converter.normalize_link(href)})"

    def _render_bold(self, element) -> str:
        """Render ``<b>`` / ``<strong>`` as bold."""
        return f"**{self._render_inline(element)}**"

    def _render_italic(self, element) -> str:
        """Render ``<i>`` / ``<em>`` as italic."""
        return f"*{self._render_inline(element)}*"

    def _render_underline(self, element) -> str:
        """Render ``<u>`` as raw HTML, Markdown having no underline."""
        return f"<u>{self._render_inline(element)}</u>"

    def _render_inline_code(self, element) -> str:
        """Render ``<code>`` as a code span, keeping only its text content."""
        return f"`{element.text_content()}`"

    def _render_break(self, element) -> str:
        """Render ``<br>`` as raw HTML.

        A Markdown hard break (two trailing spaces or a backslash) is fragile:
        trailing whitespace gets stripped, and a real newline breaks list-item
        continuation and blockquote lines instead of staying inside them. Raw
        ``<br>`` survives in every inline context — paragraph, heading, list
        item, blockquote and table cell — and this module already emits raw
        HTML for ``<u>`` and ``<video>``.

        :returns: The ``"<br>"`` tag.
        """
        return "<br>"

    def _render_reference(self, element) -> str:
        """Render a Zhihu citation (``sup[data-draft-type="reference"]``).

        The marker carries its target URL and, when the author supplied one, a
        title; both are recorded for :meth:`_append_references` rather than
        rendered inline.

        :param element: The citation ``<sup>`` element.
        :returns: A ``[^n]`` footnote marker, or the bare marker text when the
            citation has no URL to define it with.
        """
        number = (element.get("data-numero") or "").strip() or element.text_content().strip().strip("[]")
        target = (element.get("data-url") or "").strip()
        if not number or not target:
            return element.text_content().strip()
        self._references.setdefault(number, ((element.get("data-text") or "").strip(), target))
        return f"[^{number}]"

    def _render_math(self, element) -> str:
        """Render a Zhihu formula span (``span.ztext-math``) as LaTeX."""
        tex = element.get("data-tex", "")
        eeimg = element.get("data-eeimg", "")  # Zhihu formula type identifier

        if not tex:
            return element.text_content()
        # data-eeimg="2" means block/display formula
        if eeimg == "2" or "\\tag" in tex:
            return f"\n$$\n{tex}\n$$\n"
        return f"${tex}$"

    def _render_video(self, element) -> str | None:
        """Render a Zhihu video wrapper.

        :returns: The video tag, or the generic child rendering when ``src`` is empty.
        """
        video = element.find(".//video")
        if video is not None:
            src = video.get("src", "")
            if src:
                return f'<video src="{src}"></video>'
        return self._render_children(element)

    # ── generic fallback ─────────────────────────────────────────────────

    def _render_children(self, element) -> str | None:
        """Recursively render an element's children.

        Block containers (:data:`_BLOCK_TAGS`) that hold a direct block-level
        child are a sequence of blocks, so their rendered parts join with a
        blank line. Everything else — inline containers such as ``<span>`` and
        block containers that only hold inline content — goes through
        :meth:`_render_inline`, which preserves the source whitespace instead
        of inventing spaces between adjacent inline elements.

        :param element: The element whose children should be rendered.
        :returns: The joined Markdown, or ``None`` when no child contributes.
        """
        if _tag_name(element) in _BLOCK_TAGS and _has_block_child(element):
            parts = [md for node in _iter_nodes(element) if (md := self._dispatch(node))]
            return "\n\n".join(parts) if parts else None
        return self._render_inline(element) or None

    # ── table ────────────────────────────────────────────────────────────

    def _iter_own_rows(self, table: HtmlElement):
        """Iterate the ``<tr>`` elements belonging to *table*.

        Rows of a nested table belong to that table, not to this one. The
        ancestry check walks the tree in Python once per row, so it is skipped
        altogether when *table* holds no nested table.

        :param table: The table to walk.
        :returns: An iterator over this table's rows, in document order.
        """
        if not table.findall(".//table"):
            yield from table.iter("tr")
            return
        for tr in table.iter("tr"):
            if next(tr.iterancestors("table"), None) is table:
                yield tr

    def _row_cells(self, tr: HtmlElement) -> list[str]:
        """Return the rendered cell Markdown of *tr* in document order.

        Cells are rendered inline instead of taken through ``text_content()``:
        Zhihu strips most formatting inside table cells but keeps ``<br>``, so
        ``打开文件<br>读取内容`` must stay on two lines. A real newline would
        terminate the Markdown table row, hence the ``<br>`` fallback; literal
        pipes are escaped because GFM uses them as column separators.

        :param tr: A table row.
        :returns: The Markdown of each direct ``<td>`` / ``<th>`` child.
        """
        cells: list[str] = []
        for cell in tr.xpath("./td | ./th"):
            md = self._render_inline(cell)
            # A real newline would terminate the Markdown table row.
            md = md.replace("\n", "<br>")
            # GFM tables need literal pipes escaped.
            md = md.replace("|", "\\|")
            cells.append(md.strip())
        return cells

    def _render_table(self, table: HtmlElement) -> str:
        """Convert an HTML table to a Markdown table.

        :param table: The ``<table>`` element.
        :returns: The Markdown table, or ``""`` when the table has no rows.
        """
        own_rows = list(self._iter_own_rows(table))
        if not own_rows:
            return ""

        thead = table.find("thead")
        if thead is None:
            # No explicit header: the first row doubles as one.
            header_rows, body_rows = own_rows[:1], own_rows[1:]
        else:
            is_head = [next(tr.iterancestors("thead"), None) is thead for tr in own_rows]
            header_rows = [tr for tr, head in zip(own_rows, is_head) if head]
            body_rows = [tr for tr, head in zip(own_rows, is_head) if not head]

        rows = [cells for tr in header_rows + body_rows if (cells := self._row_cells(tr))]
        if not rows:
            return ""

        # Determine column count
        max_cols = max(len(row) for row in rows)

        # Normalize rows to same column count
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # Build Markdown table
        markdown = [
            # Header row
            "| " + " | ".join(rows[0]) + " |",
            # Separator row
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        # Data rows
        markdown.extend("| " + " | ".join(row) + " |" for row in rows[1:])

        return "\n".join(markdown)


# ── dispatch tables ───────────────────────────────────────────────────────────

#: Elements that never contribute content.
_SKIP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "svg", "button", "input", "form", "nav", "header", "footer", "aside"}
)

#: Containers whose children are joined by a blank line instead of a space.
_BLOCK_TAGS: frozenset[str] = frozenset({"div", "section", "article", "main"})

#: Tags that occupy their own block when they appear as a direct child.
_BLOCK_LEVEL_TAGS: frozenset[str] = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "ul",
        "ol",
        "blockquote",
        "pre",
        "figure",
        "hr",
    }
)


def _has_block_child(element: HtmlElement) -> bool:
    """Return whether *element* has a direct block-level child element.

    Only direct children count: a ``<div>`` wrapping a paragraph is a sequence
    of blocks, while a ``<div>`` whose only block-level descendants sit inside
    a ``<span>`` is still inline content.

    :param element: The container to inspect.
    :returns: ``True`` when a direct child's tag is in :data:`_BLOCK_LEVEL_TAGS`.
    """
    return any(_tag_name(child) in _BLOCK_LEVEL_TAGS for child in element)


def _is_link_card(element: HtmlElement) -> bool:
    """Return whether *element* is a Zhihu link card."""
    return "RichText-LinkCardContainer" in element.classes


def _is_discarded_link(element: HtmlElement) -> bool:
    """Return whether *element* is a Zhihu ad or paid-consult card to drop."""
    return (
        element.get("data-draft-type") in ("ad-link-card", "edu-card")
        or element.get("data-ad-id") is not None
        or element.get("data-edu-card-id") is not None
    )


def _is_reference_sup(element: HtmlElement) -> bool:
    """Return whether *element* is a Zhihu citation marker.

    Zhihu also uses ``<sup>`` for ordinary superscripts, which the caller's tag
    guard lets fall through to the generic handler.
    """
    return element.get("data-draft-type") == "reference"


def _is_math_span(element: HtmlElement) -> bool:
    """Return whether *element* is a Zhihu formula span."""
    return "ztext-math" in element.classes


def _is_video_div(element: HtmlElement) -> bool:
    """Return whether *element* wraps a video.

    The caller guards on the tag first, so this descendant query only runs for
    ``div`` elements.
    """
    return element.find(".//video") is not None


#: Tag → handler. The values are plain functions, so the dispatcher passes
#: ``self`` explicitly. Tags that also have attribute-conditional variants are
#: guarded in :meth:`PageToMarkdown._dispatch` before this table is consulted.
_TAG_HANDLERS: dict[str, Callable[[PageToMarkdown, Any], str | None]] = {
    "h1": PageToMarkdown._render_heading,
    "h2": PageToMarkdown._render_heading,
    "h3": PageToMarkdown._render_heading,
    "h4": PageToMarkdown._render_heading,
    "h5": PageToMarkdown._render_heading,
    "h6": PageToMarkdown._render_heading,
    "p": PageToMarkdown._render_paragraph,
    "ul": PageToMarkdown._render_unordered_list,
    "ol": PageToMarkdown._render_ordered_list,
    "blockquote": PageToMarkdown._render_blockquote,
    "pre": PageToMarkdown._render_code_block,
    "hr": PageToMarkdown._render_rule,
    "img": PageToMarkdown._render_image,
    "figure": PageToMarkdown._render_figure,
    "table": PageToMarkdown._render_table,
    "a": PageToMarkdown._render_link,
    "b": PageToMarkdown._render_bold,
    "strong": PageToMarkdown._render_bold,
    "i": PageToMarkdown._render_italic,
    "em": PageToMarkdown._render_italic,
    "u": PageToMarkdown._render_underline,
    "code": PageToMarkdown._render_inline_code,
    "br": PageToMarkdown._render_break,
    "div": PageToMarkdown._render_children,
    "section": PageToMarkdown._render_children,
    "article": PageToMarkdown._render_children,
    "main": PageToMarkdown._render_children,
    "span": PageToMarkdown._render_children,
}


def calculate_text_length(html_content: str) -> int:
    """Return the length of visible text in *html_content*, excluding markup."""
    if not html_content:
        return 0

    doc = _get_lxml_html().fromstring(html_content)

    # self:: covers root element for single-img fragments
    for img in doc.xpath(".//img | self::img"):
        cls = img.get("class", "")
        if _eeimg_re.search(cls):
            if img is doc:
                return 0  # root img with eeimg class replaced by space
            replace_with_text(img, " ")
        else:
            img.drop_tree()

    return len(doc.text_content())


converter = PageToMarkdown()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        html_file = sys.argv[1]
        with open(html_file, encoding="utf-8") as f:
            html_content = f.read()
        print(converter.convert(html_content))
    else:
        print("Usage: python html2markdown.py <html_file>")
