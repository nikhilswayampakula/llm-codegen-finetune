from src.build_dataset import split_into_examples

def test_split_examples_python():
    code = "def add(a,b):\n    return a+b\n\nclass X:\n    pass\n"
    pairs = split_into_examples(code, '.py')
    assert len(pairs) >= 1 and 'prompt' in pairs[0] and 'completion' in pairs[0]
