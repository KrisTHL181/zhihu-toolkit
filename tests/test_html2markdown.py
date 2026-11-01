"""Tests for the HTML → Markdown inline/whitespace and ``<br>`` handling."""

from __future__ import annotations

import pytest

from zhihu_cli.content.utils.html2markdown import PageToMarkdown


@pytest.fixture
def converter() -> PageToMarkdown:
    """Return a fresh converter for each test."""
    return PageToMarkdown()


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p>1<br>2<br>3</p>", "1<br>2<br>3"),
        ("<p>a<br><br>b</p>", "a<br><br>b"),
        ("<div>a<br>b</div>", "a<br>b"),
    ],
)
def test_break_becomes_raw_html(converter: PageToMarkdown, html: str, expected: str) -> None:
    """``<br>`` survives as raw HTML instead of a fragile Markdown newline."""
    assert converter.convert(html) == expected


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p>这是<strong>重点</strong>内容</p>", "这是**重点**内容"),
        ("<p>Hello <strong>world</strong>!</p>", "Hello **world**!"),
        ("<p>a<span>b</span>c</p>", "abc"),
        ('<p>参见<a href="https://example.com">链接</a>。</p>', "参见[链接](https://example.com)。"),
    ],
)
def test_inline_whitespace_is_preserved(converter: PageToMarkdown, html: str, expected: str) -> None:
    """No spaces are inserted around adjacent inline elements."""
    assert converter.convert(html) == expected


def test_heading_span_does_not_gain_space(converter: PageToMarkdown) -> None:
    """An SSR entity span must not split the heading text."""
    html = '<h2><span data-search-entity="0">化核二项回文理论</span>与半回文理论</h2>'

    assert converter.convert(html) == "## 化核二项回文理论与半回文理论"


def test_html_whitespace_collapses_to_one_space(converter: PageToMarkdown) -> None:
    """Runs of HTML whitespace inside a text node become a single space."""
    assert converter.convert("<p>a\n\t  b</p>") == "a b"


def test_empty_paragraph_is_skipped(converter: PageToMarkdown) -> None:
    """A paragraph holding only a ``<br>`` contributes nothing."""
    assert converter.convert('<p class="ztext-empty-paragraph"><br></p>') == ""


def test_adjacent_paragraphs_keep_blank_line(converter: PageToMarkdown) -> None:
    """The block container still joins its block children with a blank line."""
    assert converter.convert("<p>1</p><p>2</p>") == "1\n\n2"


def test_table_output_unchanged(converter: PageToMarkdown) -> None:
    """Simple tables stay byte-identical."""
    assert converter.convert("<table><tr><td>a</td><td>b</td></tr></table>") == "| a | b |\n| --- | --- |"


def test_table_cell_preserves_break(converter: PageToMarkdown) -> None:
    """``<br>`` inside a cell survives instead of being flattened."""
    html = "<table><tr><th>步骤</th><th>描述</th></tr><tr><td>第一步</td><td>打开文件<br>读取内容</td></tr></table>"

    assert converter.convert(html) == ("| 步骤 | 描述 |\n| --- | --- |\n| 第一步 | 打开文件<br>读取内容 |")


def test_table_cell_list_like_break(converter: PageToMarkdown) -> None:
    """A list expressed with ``<br>`` stays inside its cell."""
    html = "<table><tr><td>- 苹果<br>- 香蕉<br>- 橙子</td></tr></table>"

    assert converter.convert(html) == "| - 苹果<br>- 香蕉<br>- 橙子 |\n| --- |"


def test_table_cell_escapes_pipe(converter: PageToMarkdown) -> None:
    """A literal pipe in a cell is escaped so it does not split the column."""
    html = "<table><tr><td>a|b</td><td>c</td></tr></table>"

    assert converter.convert(html) == "| a\\|b | c |\n| --- | --- |"


def test_table_cell_renders_inline_markup(converter: PageToMarkdown) -> None:
    """Inline markup in arbitrary HTML cells still renders."""
    html = "<table><tr><td><strong>x</strong></td></tr></table>"

    assert converter.convert(html) == "| **x** |\n| --- |"


def test_unordered_list_unchanged(converter: PageToMarkdown) -> None:
    """Bullet lists render as before."""
    assert converter.convert("<ul><li>a</li><li>b</li></ul>") == "- a\n- b"


def test_blockquote_unchanged(converter: PageToMarkdown) -> None:
    """Blockquotes render as before."""
    assert converter.convert("<blockquote>引用</blockquote>") == "> 引用"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p><em>a</em><em>b</em></p>", "*ab*"),
        ("<p><strong>a</strong><strong>b</strong></p>", "**ab**"),
        ("<p><code>a</code><code>b</code></p>", "`ab`"),
        ("<p><strong>a</strong><strong>b</strong><strong>c</strong></p>", "**abc**"),
    ],
)
def test_adjacent_same_type_elements_merge(converter: PageToMarkdown, html: str, expected: str) -> None:
    """Adjacent same-type elements merge into one unambiguous delimiter run."""
    assert converter.convert(html) == expected


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p><em>a</em><strong>b</strong></p>", "*a***b**"),
        ("<p><strong>a</strong><em>b</em></p>", "**a***b*"),
        ("<p><em>a</em><strong>b</strong><em>c</em></p>", "*a***b**<!-- -->*c*"),
    ],
)
def test_simple_mixed_emphasis_concatenates(converter: PageToMarkdown, html: str, expected: str) -> None:
    """Simple mixed emphasis is unambiguous and needs no separator."""
    assert converter.convert(html) == expected


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p><em>a</em><em><strong>b</strong></em></p>", "*a*<!-- -->***b***"),
        ("<p><strong>a</strong><strong><em>b</em></strong></p>", "**a**<!-- -->***b***"),
        (
            "<p><em><strong>a</strong></em><em><strong>b</strong></em></p>",
            "***a***<!-- -->***b***",
        ),
    ],
)
def test_unsafe_adjacent_elements_keep_separator(converter: PageToMarkdown, html: str, expected: str) -> None:
    """Nested delimiters fall back to an invisible comment."""
    assert converter.convert(html) == expected


def test_text_separates_adjacent_elements(converter: PageToMarkdown) -> None:
    """A whitespace text node is a separator; no comment is needed."""
    assert converter.convert("<p><strong>a</strong> <strong>b</strong></p>") == "**a** **b**"


def test_links_need_no_separator(converter: PageToMarkdown) -> None:
    """Link output ends with ``)``, so adjacent links do not merge."""
    assert converter.convert('<p><a href="x">a</a><a href="y">b</a></p>') == "[a](x)[b](y)"


def test_separator_renders_as_distinct_spans() -> None:
    """markdown-it parses the separated runs as separate inline elements."""
    markdown_it = pytest.importorskip("markdown_it")
    render = markdown_it.MarkdownIt().render

    assert render("**a**<!-- -->**b**") == "<p><strong>a</strong><!-- --><strong>b</strong></p>\n"
    assert render("*a*<!-- -->*b*") == "<p><em>a</em><!-- --><em>b</em></p>\n"
    assert render("`a`<!-- -->`b`") == "<p><code>a</code><!-- --><code>b</code></p>\n"


def test_merged_run_renders_as_one_element() -> None:
    """A merged run stays a single strong element, and mixed emphasis is clean."""
    markdown_it = pytest.importorskip("markdown_it")
    render = markdown_it.MarkdownIt().render

    assert render("**ab**") == "<p><strong>ab</strong></p>\n"
    assert render("*a***b**") == "<p><em>a</em><strong>b</strong></p>\n"


def test_nested_runs_render_as_separate_spans() -> None:
    """The fallback keeps two nested em+strong runs distinct."""
    markdown_it = pytest.importorskip("markdown_it")

    assert (
        markdown_it.MarkdownIt().render("***a***<!-- -->***b***")
        == "<p><em><strong>a</strong></em><!-- --><em><strong>b</strong></em></p>\n"
    )
