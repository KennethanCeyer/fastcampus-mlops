import pytest
from add import add

@pytest.mark.parametrize(
   "given, expected",
   [
       ((1, 3), 4),
       ((2, 7), 9),
       ((-1, 4), 3),
       ((0, 0), 0),
   ]
)
def test_add(given: tuple[int, int], expected: int):
   result = add(*given)
   assert result == expected

