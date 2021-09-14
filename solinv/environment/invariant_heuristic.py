from ..tyrell.dsl import Node, HoleNode
from ..tyrell.spec import FunctionProduction

class InvariantHeuristic():
    '''
    Provides a set of invariant heuristic checkings.
    Returns True if the given heuristic is good to go, False if not.
    '''

    def __init__():
        pass

    # @classmethod
    # def no_duplicate_children(self, arg_prods, arg_inv):
    #     '''
    #     Check if *all* children of nodes of designated production rules are exactly the same.
    #     '''
    #     assert len(arg_prods)>0, "Should provide at least one production rule for checking."
    #     if isinstance(arg_inv, HoleNode):
    #         # if root node is a hole, just return False
    #         return True
    #     elif arg_inv.production in arg_prods:
    #         sig_list = [str(p) for p in arg_inv.children]
    #         sig_set = set(sig_list)
    #         return len(sig_set) > 1
    #     else:
    #         # move on to next level
    #         res_list = [InvariantHeuristic.no_duplicate_children(p) for p in arg_inv.children]
    #         return all(res_list)

    @classmethod
    def no_duplicate_children(self, arg_inv):
        '''
        Check if *all* children of nodes look the same.
        Apply to complete invariant.
        '''
        if isinstance(arg_inv, HoleNode):
            # if root node is a hole, just return False
            return True
        elif len(arg_inv.children) > 1:
            sig_list = [str(p) for p in arg_inv.children]
            sig_set = set(sig_list)
            return len(sig_set) > 1
        else:
            # move on to next level
            res_list = [InvariantHeuristic.no_duplicate_children(p) for p in arg_inv.children]
            return all(res_list)

    @classmethod
    def no_enum2expr_root(self, arg_inv):
        '''
        Check if the root node is enum2expr.
        Apply to complete and partial invariants.
        '''
        if isinstance(arg_inv, HoleNode):
            return True
        elif isinstance(arg_inv.production, FunctionProduction):
            if arg_inv.production.name == "enum2expr":
                return False
            else:
                return True
        else:
            return True
