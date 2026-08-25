"""Tests for the chunking strategies and the similarity maths.

These need no model and no Ollama: splitting text and computing a cosine are
pure functions. Run:  python -m pytest tests.py -q
"""

import math
import unittest

from chunking import TEXT, fixed_size_chunks, hierarchical_chunks, semantic_chunks
from embeddings import cosine_similarity, search


class TestFixedSize(unittest.TestCase):
    def test_every_chunk_is_at_most_the_requested_size(self):
        for chunk in fixed_size_chunks(TEXT, 50):
            self.assertLessEqual(len(chunk), 50)

    def test_nothing_is_lost_or_duplicated(self):
        self.assertEqual("".join(fixed_size_chunks(TEXT, 50)), TEXT)

    def test_the_expected_number_of_chunks(self):
        self.assertEqual(len(fixed_size_chunks(TEXT, 50)), math.ceil(len(TEXT) / 50))

    def test_it_cuts_words_in_half(self):
        """The weakness this strategy is included to demonstrate."""
        chunks = fixed_size_chunks(TEXT, 50)
        # 'automate' is split across chunks 2 and 3.
        self.assertTrue(chunks[1].endswith("automa"))
        self.assertTrue(chunks[2].startswith("te "))


class TestSemantic(unittest.TestCase):
    def test_three_sentences_become_three_chunks(self):
        self.assertEqual(len(semantic_chunks(TEXT)), 3)

    def test_every_chunk_ends_a_sentence(self):
        for chunk in semantic_chunks(TEXT):
            self.assertTrue(chunk.endswith((".", "!", "?")), chunk)

    def test_no_word_is_broken(self):
        for chunk in semantic_chunks(TEXT):
            self.assertIn(chunk.rstrip("."), TEXT)

    def test_the_ethics_sentence_stays_whole(self):
        """The sentence the Step 6 query is looking for must not be split."""
        chunks = semantic_chunks(TEXT)
        ethics = [c for c in chunks if "ethical" in c]
        self.assertEqual(len(ethics), 1)
        for word in ("privacy", "transparency", "employment"):
            self.assertIn(word, ethics[0])


class TestHierarchical(unittest.TestCase):
    def test_it_returns_a_tree(self):
        tree = hierarchical_chunks(TEXT)
        self.assertEqual(len(tree), 1)               # one paragraph
        self.assertIn("children", tree[0])
        self.assertGreater(len(tree[0]["children"]), 1)

    def test_children_respect_the_size_limit(self):
        for node in hierarchical_chunks(TEXT, max_chars=90):
            for child in node["children"]:
                self.assertLessEqual(len(child), 90, child)

    def test_several_paragraphs_become_several_branches(self):
        document = "First para sentence one. Second sentence.\n\nSecond para here. And more."
        tree = hierarchical_chunks(document, max_chars=40)
        self.assertEqual(len(tree), 2)

    def test_a_short_paragraph_is_not_split(self):
        tree = hierarchical_chunks("Short enough.", max_chars=90)
        self.assertEqual(tree[0]["children"], ["Short enough."])

    def test_children_come_from_the_source(self):
        for node in hierarchical_chunks(TEXT):
            for child in node["children"]:
                self.assertIn(child.strip(".,"), TEXT)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors_score_minus_one(self):
        self.assertAlmostEqual(cosine_similarity([1, 1], [-1, -1]), -1.0)

    def test_length_does_not_matter_only_direction(self):
        """Why cosine and not Euclidean distance: a long chunk and a short chunk
        about the same topic point the same way."""
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [10, 20, 30]), 1.0)

    def test_a_zero_vector_gives_zero_not_a_crash(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_mismatched_dimensions_raise(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.store = [
            {"id": 1, "text": "far", "embedding": [0, 1], "chars": 3},
            {"id": 2, "text": "exact", "embedding": [1, 0], "chars": 5},
            {"id": 3, "text": "near", "embedding": [1, 0.5], "chars": 4},
        ]

    def test_results_are_sorted_descending(self):
        results = search([1, 0], self.store, top_k=3)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_the_best_match_is_first(self):
        self.assertEqual(search([1, 0], self.store, top_k=1)[0]["text"], "exact")

    def test_top_k_limits_the_results(self):
        self.assertEqual(len(search([1, 0], self.store, top_k=2)), 2)

    def test_each_score_stays_attached_to_its_chunk(self):
        for result in search([1, 0], self.store, top_k=3):
            self.assertIn("text", result)
            self.assertIn("score", result)
            self.assertIn("embedding", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
