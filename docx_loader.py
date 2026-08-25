"""Load a .docx into structured blocks, using only the standard library.

A .docx is a zip archive of XML. `word/document.xml` holds the body, where each
paragraph is a <w:p> element and its visible text sits in <w:t> runs. Reading it
directly avoids a python-docx dependency and, more importantly, keeps the
information the structure-aware chunker needs: which paragraphs are headings,
which are list items, and where the section boundaries fall.

Plain text extraction would throw all of that away — everything would arrive as
one undifferentiated blob, and a heading would be indistinguishable from a
sentence.
"""

import html
import re
import zipfile
from dataclasses import dataclass, field
from typing import List


@dataclass
class Block:
    """One paragraph of the document, with the structure Word recorded."""

    text: str
    style: str = ""           # Word style name: Title, Heading1, ListParagraph...
    is_heading: bool = False
    heading_level: int = 0    # 0 = not a heading, 1 = Title/Heading1, 2 = Heading2
    is_list_item: bool = False

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class Document:
    """A loaded document: the blocks, plus a few convenience views."""

    blocks: List[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Everything as one string, paragraphs separated by blank lines.

        Blank-line separation matters: the recursive chunker's strongest
        separator is "\\n\\n", so joining with a single newline would hide every
        paragraph boundary from it.
        """
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def chars(self) -> int:
        return len(self.text)

    def sections(self) -> List[dict]:
        """Group blocks under the heading that precedes them.

        Returns [{"title": str, "level": int, "blocks": [Block, ...]}, ...].
        This is the document's own outline, which the structure-aware and
        hierarchical strategies both build on.
        """
        sections: List[dict] = []
        current = {"title": "(front matter)", "level": 0, "blocks": []}

        for block in self.blocks:
            if block.is_heading:
                # Close the previous section before opening the next.
                if current["blocks"] or current["title"] != "(front matter)":
                    sections.append(current)
                current = {"title": block.text, "level": block.heading_level, "blocks": []}
            else:
                current["blocks"].append(block)

        sections.append(current)
        return [section for section in sections if section["blocks"]]


def _style_of(paragraph_xml: str) -> str:
    match = re.search(r'w:pStyle w:val="([^"]+)"', paragraph_xml)
    return match.group(1) if match else ""


def _text_of(paragraph_xml: str) -> str:
    """Join the <w:t> runs of one paragraph into a single string.

    Word splits a sentence across several runs whenever formatting changes, so
    "the **bold** word" arrives as three runs and has to be reassembled.
    html.unescape turns &amp; and &quot; back into real characters.
    """
    runs = re.findall(r"<w:t[^>]*>(.*?)</w:t>", paragraph_xml, re.S)
    text = "".join(runs)
    text = re.sub(r"<[^>]+>", "", text)          # drop any stray inline tags
    return html.unescape(text).strip()


def load_docx(path: str) -> Document:
    """Read a .docx and return its blocks in document order."""
    try:
        archive = zipfile.ZipFile(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such document: {path}")
    except zipfile.BadZipFile:
        raise ValueError(f"{path} is not a valid .docx (not a zip archive)")

    with archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError(f"{path} has no word/document.xml — is it really a .docx?")
        xml = archive.read("word/document.xml").decode("utf-8")

    blocks = []
    # Non-greedy match per <w:p> element, in document order.
    for match in re.finditer(r"<w:p[ >].*?</w:p>", xml, re.S):
        paragraph_xml = match.group(0)
        text = _text_of(paragraph_xml)

        if not text:
            continue  # empty paragraphs are layout spacing, not content

        style = _style_of(paragraph_xml)

        # Word records list membership as <w:numPr>, separately from the style.
        is_list_item = "<w:numPr" in paragraph_xml or style == "ListParagraph"

        heading_level = 0
        if style in ("Title", "Heading1"):
            heading_level = 1
        elif style.startswith("Heading"):
            digits = re.sub(r"\D", "", style)
            heading_level = int(digits) if digits else 2

        blocks.append(
            Block(
                text=text,
                style=style,
                is_heading=heading_level > 0,
                heading_level=heading_level,
                is_list_item=is_list_item,
            )
        )

    return Document(blocks=blocks)


def describe(document: Document) -> str:
    """A short profile of what was loaded."""
    headings = [b for b in document.blocks if b.is_heading]
    lists = [b for b in document.blocks if b.is_list_item]
    prose = [b for b in document.blocks if not b.is_heading and not b.is_list_item]

    lines = [
        f"blocks        : {len(document.blocks)}",
        f"  headings    : {len(headings)}",
        f"  list items  : {len(lists)}",
        f"  prose paras : {len(prose)}",
        f"characters    : {document.chars:,}",
        f"sections      : {len(document.sections())}",
    ]
    return "\n".join(lines)
