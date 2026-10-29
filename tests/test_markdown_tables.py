"""Assistant-message markdown tables render as boxed tables."""

from __future__ import annotations

import unittest

from xarness.tui import flow


def _render(markdown: str, width: int = 60) -> list[str]:
    return [strip.text.rstrip() for strip in flow._settled_strips(markdown, width)]


class MarkdownHeadingTest(unittest.TestCase):
    def test_h1_is_left_aligned_with_a_blank_line_above(self) -> None:
        rows = _render("Intro\n\n# Title\n\nbody\n")

        index = rows.index("Title")
        self.assertEqual(rows[index - 1], "")
        self.assertFalse(rows[index].startswith(" "))

    def test_leading_heading_gets_no_top_padding(self) -> None:
        rows = _render("# Title\n\nbody\n")

        self.assertEqual(rows[0], "Title")

    def test_heading_opening_a_later_chunk_gets_top_padding(self) -> None:
        rows = [
            strip.text.rstrip() for strip in flow._settled_strips(
                "# Title\n\nbody\n", 60, pad_leading=True,
            )
        ]

        self.assertEqual(rows[0], "")
        self.assertEqual(rows[1], "Title")

    def test_every_heading_level_gets_top_padding_after_content(self) -> None:
        markdown = "intro\n\n" + "\n\n".join(
            f"{'#' * level} Level {level}" for level in range(1, 7)
        )
        rows = _render(markdown)

        for level in range(1, 7):
            index = rows.index(f"Level {level}")
            self.assertEqual(rows[index - 1], "")
            self.assertFalse(rows[index].startswith(" "))


class MarkdownTableTest(unittest.TestCase):
    def test_table_is_rendered_with_borders(self) -> None:
        rows = _render(
            "| Name | Type |\n"
            "|------|------|\n"
            "| a    | int  |\n"
            "| b    | str  |\n"
        )

        self.assertTrue(rows[0].startswith("╭"))
        self.assertTrue(rows[-1].startswith("╰"))
        # Header separator between the header and the body rows.
        self.assertTrue(any(row.startswith("├") for row in rows))
        # Every row line is boxed: vertical border on both edges.
        for row in rows[1:-1]:
            if row.startswith("├"):
                continue
            self.assertTrue(row.startswith("│"), row)
            self.assertTrue(row.endswith("│"), row)

        text = "\n".join(rows)
        for cell in ("Name", "Type", "a", "int", "b", "str"):
            self.assertIn(cell, text)

    def test_table_leaves_the_final_column_blank(self) -> None:
        """Don't end a row on the terminal's last column: some terminals keep
        that cell in a pending-wrap state and drop the glyph (the right
        border) when a row wraps."""
        width = 40
        rows = [
            strip.text for strip in flow._settled_strips(
                "| A | B |\n|---|---|\n| a long value that wraps | 2 |\n",
                width,
            )
        ]
        table_rows = [
            row for row in rows if row.startswith(("╭", "│", "├", "╰"))
        ]
        self.assertTrue(table_rows)
        for row in table_rows:
            self.assertEqual(len(row), width)
            self.assertEqual(row[-1], " ")
            self.assertIn(row.rstrip()[-1], "│╮╯┤")

    def test_every_body_row_is_separated_by_a_line(self) -> None:
        rows = _render(
            "| H |\n"
            "|---|\n"
            "| a |\n"
            "| b |\n"
            "| c |\n"
        )

        separators = [row for row in rows if row.startswith("├")]
        self.assertEqual(len(separators), 3)
        self.assertEqual(rows.count("├───┤"), 3)

    def test_table_survives_blank_strip_trimming(self) -> None:
        """Prose rendering drops blank strips; the table borders must stay."""
        rows = _render(
            "Intro\n\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n\n"
            "Outro\n"
        )

        table = [row for row in rows if row.startswith(("╭", "│", "├", "╰"))]
        self.assertEqual(len(table), 5)
        self.assertIn("Intro", rows)
        self.assertIn("Outro", rows)

    def test_narrow_table_wraps_instead_of_ellipsizing(self) -> None:
        rows = _render(
            "| Name | Type | Notes |\n"
            "|------|------|-------|\n"
            "| alpha | integer | first item |\n"
            "| beta  | string  | second item |\n",
            width=20,
        )

        drawn = "\n".join(rows)
        self.assertNotIn("…", drawn)
        # Strip borders/whitespace: wrapping may split words, but nothing may
        # be dropped.
        content = "".join(
            ch for ch in drawn
            if not ch.isspace() and ch not in "╭╮╰╯├┤┬┴┼─│"
        )
        # Column-width wrapping interleaves cells across lines, so compare
        # the characters as a multiset rather than a contiguous string.
        self.assertEqual(
            sorted(content),
            sorted("NameTypeNotesalphaintegerfirstitembetastringseconditem"),
        )

    def test_wide_cells_wrap_inside_the_border(self) -> None:
        rows = _render(
            "| A | Notes |\n"
            "|---|-------|\n"
            "| 1 | a long note that should wrap across lines |\n",
            width=30,
        )

        self.assertTrue(rows[0].startswith("╭"))
        self.assertTrue(rows[-1].startswith("╰"))
        wrapped = " ".join(" ".join(row.strip("│ ").split()) for row in rows)
        self.assertIn("a long note that should wrap across lines", wrapped)
