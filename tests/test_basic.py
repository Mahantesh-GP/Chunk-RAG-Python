import os
from rag_eval.strategies import FixedSizeChunker, StructureChunker, MultigranularChunker


def test_fixed_chunker():
    text = "word " * 600
    c = FixedSizeChunker(chunk_size=256, chunk_overlap=20)
    chunks = c.chunk(text)
    assert len(chunks) >= 2


def test_structure_chunker():
    text = "\n\n".join(["Paragraph " + str(i) for i in range(20)])
    c = StructureChunker(min_chunk_size=10, max_chunk_size=50)
    chunks = c.chunk(text)
    assert len(chunks) >= 1


def test_multigranular():
    text = "word " * 1200
    c = MultigranularChunker()
    chunks = c.chunk(text)
    assert len(chunks) >= 1


if __name__ == '__main__':
    test_fixed_chunker()
    test_structure_chunker()
    test_multigranular()
    print('All basic tests passed')
