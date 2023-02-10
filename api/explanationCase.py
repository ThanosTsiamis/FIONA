from enum import Enum


class Codes(Enum):
    A = ""
    B = ""
    C = ""
    AB = ""
    AC = ""
    BC = ""
    ABC = ""
    FUNCTIONAL_DEPENDENCY = ""


if __name__ == "__main__":
    print(Codes.TESTING.value)
