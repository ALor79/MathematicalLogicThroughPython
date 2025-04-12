# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/syntax.py

"""Syntactic handling of propositional formulas."""

from __future__ import annotations
from functools import lru_cache
from typing import Mapping, Optional, Set, Tuple, Union

from logic_utils import frozen, memoized_parameterless_method

@lru_cache(maxsize=100) # Cache the return value of is_variable
def is_variable(string: str) -> bool:
    """Checks if the given string is a variable name.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a variable name, ``False`` otherwise.
    """
    return string[0] >= 'p' and string[0] <= 'z' and \
           (len(string) == 1 or string[1:].isdecimal())

@lru_cache(maxsize=100) # Cache the return value of is_constant
def is_constant(string: str) -> bool:
    """Checks if the given string is a constant.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a constant, ``False`` otherwise.
    """
    return string == 'T' or string == 'F'

@lru_cache(maxsize=100) # Cache the return value of is_unary
def is_unary(string: str) -> bool:
    """Checks if the given string is a unary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a unary operator, ``False`` otherwise.
    """
    return string == '~'

@lru_cache(maxsize=100) # Cache the return value of is_binary
def is_binary(string: str) -> bool:
    """Checks if the given string is a binary operator.

    Parameters:
        string: string to check.

    Returns:
        ``True`` if the given string is a binary operator, ``False`` otherwise.
    """
    return string == '&' or string == '|' or string == '->'
    # For Chapter 3:
    # return string in {'&', '|',  '->', '+', '<->', '-&', '-|'}

@frozen
class Formula:
    """An immutable propositional formula in tree representation, composed from
    variable names, and operators applied to them.

    Attributes:
        root (`str`): the constant, variable name, or operator at the root of
            the formula tree.
        first (`~typing.Optional`\\[`Formula`]): the first operand of the root,
            if the root is a unary or binary operator.
        second (`~typing.Optional`\\[`Formula`]): the second operand of the
            root, if the root is a binary operator.
    """
    root: str
    first: Optional[Formula]
    second: Optional[Formula]

    def __init__(self, root: str, first: Optional[Formula] = None,
                 second: Optional[Formula] = None):
        """Initializes a `Formula` from its root and root operands.

        Parameters:
            root: the root for the formula tree.
            first: the first operand for the root, if the root is a unary or
                binary operator.
            second: the second operand for the root, if the root is a binary
                operator.
        """
        if is_variable(root) or is_constant(root):
            assert first is None and second is None
            self.root = root
        elif is_unary(root):
            assert first is not None and second is None
            self.root, self.first = root, first
        else:
            assert is_binary(root)
            assert first is not None and second is not None
            self.root, self.first, self.second = root, first, second

    @memoized_parameterless_method
    def __repr__(self) -> str:
        """Computes the string representation of the current formula.

        Returns:
            The standard string representation of the current formula.
        """
        
        # Task 1.1
        if is_variable(self.root) or is_constant(self.root):
            return self.root
        elif is_unary(self.root):
            return self.root+repr(self.first)
        else:  # binary operator
            return '('+repr(self.first)+self.root+repr(self.second)+')'

    def __eq__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is a `Formula` object that equals the
            current formula, ``False`` otherwise.
        """
        return isinstance(other, Formula) and str(self) == str(other)

    def __ne__(self, other: object) -> bool:
        """Compares the current formula with the given one.

        Parameters:
            other: object to compare to.

        Returns:
            ``True`` if the given object is not a `Formula` object or does not
            equal the current formula, ``False`` otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        return hash(str(self))

    @memoized_parameterless_method
    def variables(self) -> Set[str]:

        """Finds all variable names in the current formula.

        Returns:
            A set of all variable names used in the current formula.
        """
        # Task 1.2

        if is_variable(self.root):
            return {self.root}
        elif is_constant(self.root):
            return set()
        elif is_unary(self.root):
            return self.first.variables()
        else:  # binary operator
            return self.first.variables().union(self.second.variables())
        


    @memoized_parameterless_method
    def operators(self) -> Set[str]:
        """Finds all operators in the current formula.

        Returns:
            A set of all operators (including ``'T'`` and ``'F'``) used in the
            current formula.
        """
        print(self.__repr__())
        if is_constant(self.root):
            return {self.root}
        elif is_variable(self.root):
            return set()
        elif is_unary(self.root):
            return {self.root}.union(self.first.operators())
        else:  # binary operator
            return {self.root}.union(self.second.operators()).union(self.first.operators())
        
    @staticmethod
    def find_top_level_op(string: str) -> Tuple[str, int]:
        """Finds the top-level binary operator in a formula string of the form (S O S).

        Parameters:
            string: The formula string, assumed to start with '(' and end with ')'.

        Returns:
            A tuple containing the operator and its position in the string.

        Raises:
            ValueError: If no top-level binary operator is found.
        """
        depth = 0
        i = 1  # skip the first '('
        while i < len(string) - 1:  # skip the final ')'
            if string[i] == '(':
                depth += 1
                i += 1
            elif string[i] == ')':
                depth -= 1
                i += 1
            elif depth == 0:
                if string[i:i+2] == '->':
                    return '->', i
                elif string[i] in {'&', '|'}:
                    return string[i], i
                else:
                    i += 1
            else:
                i += 1

        raise ValueError(f"No top-level binary operator found in: {string}")

    @staticmethod
    def _parse_prefix(string: str) -> Tuple[Union['Formula', None], str]:
        """Parses a prefix of the given string into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A pair of the parsed formula and the unparsed suffix of the string.
            If the given string has as a prefix a variable name (e.g.,
            ``'x12'``) or a unary operator followed by a variable name, then the
            parsed prefix will include that entire variable name (and not just a
            part of it, such as ``'x1'``). If no prefix of the given string is a
            valid standard string representation of a formula then returned pair
            should be of ``None`` and an error message, where the error message
            is a string with some human-readable content.
        """
        """
            CFG:

            S → '( SOS ')      // Binary formula with parentheses and operator
            S → '~ S           // Unary formula (negation)
            S → V            // A variable
            S → 'T | 'F        // Constants (true or false)

            O → '-> | '& | '|(OR)   // Binary operators

            V → LN          // Variable = letter followed by optional digits
            L → 'p | 'q | ... | 'z
            N → ε | '0 | '1 | ... | '9 | NN
        """
        # Task 1.4
        if not string:
            return None, "Empty input"

        # Case 1: Constants 'T' or 'F'
        if string.startswith('T') or string.startswith('F'):
            if len(string) == 1 or not string[1].isalnum():
                return Formula(string[0]), string[1:]
            else:
                return None, f"Invalid constant usage: {string}"

        # Case 2: Variable (starts with letter p–z, followed by digits)
        if 'p' <= string[0] <= 'z':
            i = 1
            while i < len(string) and string[i].isdigit():
                i += 1
            var = string[:i]
            rest = string[i:]
            return Formula(var), rest

        # Case 3: Unary operator '~'
        if string.startswith('~'):
            subformula, rest = Formula._parse_prefix(string[1:])
            if subformula is None:
                return None, rest
            return Formula('~', subformula), rest


        # Case 4: Binary operator in parentheses
        if string.startswith('('):
            depth = 0
            for i in range(len(string)):
                if string[i] == '(':
                    depth += 1
                elif string[i] == ')':
                    depth -= 1
                    if depth == 0:
                        # Full binary formula is in string[:i+1]
                        subformula_str = string[:i+1]
                        remaining = string[i+1:]

                        try:
                            op, op_pos = Formula.find_top_level_op(subformula_str)

                            if op != '->':
                                left_str = subformula_str[1:op_pos]
                                right_str = subformula_str[op_pos + 1:-1]
                            else:
                                left_str = subformula_str[1:op_pos]
                                right_str = subformula_str[op_pos + 2:-1]

                            left_formula, rest1 = Formula._parse_prefix(left_str)
                            right_formula, rest2 = Formula._parse_prefix(right_str)

                            if left_formula is None:
                                return None, rest1
                            if right_formula is None:
                                return None, rest2
                            if rest1 != '' or rest2 != '':
                                return None, "Extra characters after subformulas"

                            return Formula(op, left_formula, right_formula), remaining

                        except Exception as e:
                            return None, f"Error parsing binary formula: {e}"

            return None, f"Unmatched parentheses in: {string}"


        # If none of the rules matched
        return None, f"Unrecognized formula start: {string}"



    @staticmethod
    def is_formula(string: str) -> bool:
        """Checks if the given string is a valid representation of a formula.

        Parameters:
            string: string to check.

        Returns:
            ``True`` if the given string is a valid standard string
            representation of a formula, ``False`` otherwise.
        """
        # Task 1.5
        var, rest = Formula._parse_prefix(string)
        return var is not None and rest is ''
        
    @staticmethod
    def parse(string: str) -> Formula:
        """Parses the given valid string representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose standard string representation is the given string.
        """
        assert Formula.is_formula(string)
        # Task 1.6

    def polish(self) -> str:
        """Computes the polish notation representation of the current formula.

        Returns:
            The polish notation representation of the current formula.
        """
        # Optional Task 1.7

    @staticmethod
    def parse_polish(string: str) -> Formula:
        """Parses the given polish notation representation into a formula.

        Parameters:
            string: string to parse.

        Returns:
            A formula whose polish notation representation is the given string.
        """
        # Optional Task 1.8

    def substitute_variables(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each variable name `v` that is a
        key in `substitution_map` with the formula `substitution_map[v]`.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            variable name occurrences originating in the current formula are
            substituted (i.e., variable name occurrences originating in one of
            the specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((p->p)|r)').substitute_variables(
            ...     {'p': Formula.parse('(q&r)'), 'r': Formula.parse('p')})
            (((q&r)->(q&r))|p)
        """
        for variable in substitution_map:
            assert is_variable(variable)
        # Task 3.3

    def substitute_operators(self, substitution_map: Mapping[str, Formula]) -> \
            Formula:
        """Substitutes in the current formula, each constant or operator `op`
        that is a key in `substitution_map` with the formula
        `substitution_map[op]` applied to its (zero or one or two) operands,
        where the first operand is used for every occurrence of ``'p'`` in the
        formula and the second for every occurrence of ``'q'``.

        Parameters:
            substitution_map: mapping defining the substitutions to be
                performed.

        Returns:
            The formula resulting from performing all substitutions. Only
            operator occurrences originating in the current formula are
            substituted (i.e., operator occurrences originating in one of the
            specified substitutions are not subjected to additional
            substitutions).

        Examples:
            >>> Formula.parse('((x&y)&~z)').substitute_operators(
            ...     {'&': Formula.parse('~(~p|~q)')})
            ~(~~(~x|~y)|~~z)
        """
        for operator in substitution_map:
            assert is_constant(operator) or is_unary(operator) or \
                   is_binary(operator)
            assert substitution_map[operator].variables().issubset({'p', 'q'})
        # Task 3.4
