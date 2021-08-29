import pandas as pd
import numpy as np

from typing import Tuple, List, Iterator, Any

from .. import spec as S
from ..dsl import Node, AtomNode, ParamNode, ApplyNode
from ..visitor import GenericVisitor
from . import Interpreter, PostOrderInterpreter, InterpreterError, GeneralError, EqualityAssertion, ComponentError
# from .context import Context
from ..logger import get_logger

logger = get_logger('tyrell.morpheus_interpreter')

def get_row(arg_tb):
    return arg_tb.shape[0]

def get_col(arg_tb):
    return arg_tb.shape[1]

def get_head(arg_tb0, arg_tb1):
    # arg_tb0 is the base
    head0 = set(arg_tb0.columns)
    content0 = set(arg_tb0.values.flatten().tolist())
    head1 = set(arg_tb1.columns)
    return len(head1 - head0 - content0)

def get_content(arg_tb0, arg_tb1):
    # arg_tb0 is the base
    content0 = set(arg_tb0.values.flatten().tolist())
    content1 = set(arg_tb1.values.flatten().tolist())
    return len(content1 - content0)

class MorpheusInterpreter(PostOrderInterpreter):

    def __init__(self, spec: S.TyrellSpec, *args, **kwargs):
        super(MorpheusInterpreter, self).__init__(*args, **kwargs)

        self._spec = spec

        # if you are debugging, turn this to False
        self._suppress_pandas_exception = True
        # self._suppress_pandas_exception = False

        self._colname_count = 0

        # fixme: this is used to compute the abstract values for content() and head()
        #        you will need to manually set it
        # self._example_input0 = None

        # note: a stateful variable that keeps track of interpretation combination
        #       which is connected to LineSkeletonEnumerator
        #       typical combination can be (1, 4, 1, 2, None, None)
        #       where None indicates anything and integers indicate production id
        #       LineSkeletonEnumerator will capture and utilize it to speed up enumeration
        self._current_combination = None

        # a numpy vectorize method to perform normalization for every cell value
        self._normalize_cell_value = np.vectorize(lambda x: self.normalize_cell_value(x))

        # given input and program, quickly identify output if computed before
        # use `hash_tb` to generate hash value for table, which is the first layer key
        # use str(prog) to generate hash value for program, which is the second layer key
        # fixme: currently this feature is not yet implemented
        self._value_caching = True # turn on value caching
        self._cached_outputs = {}

    def hash_tb(self, arg_tb):
        return tuple(pd.util.hash_pandas_object(arg_tb).tolist())

    def flatten_index(self, arg_tb):
        # test whether the index is default or not
        # default index will be something like: FrozenList([None])
        arg_drop = all(p is None for p in arg_tb.index.names)
        d0 = arg_tb.reset_index(drop=arg_drop).to_dict()
        d1 = {}
        for k,v in d0.items():
            if isinstance(k, tuple):
                k0 = [p for p in k if len(p)>0][-1]
                d1[k0] = v
            else:
                d1[k] = v
        return pd.DataFrame.from_dict(d1)

    def fresh_colname(self):
        self._colname_count += 1
        return "COL{}".format(self._colname_count-1)

    # note: do NOT call me directly, call self._normalize_cell_value
    def normalize_cell_value(self, v):
        if v is None:
            return "None"
        elif isinstance(v, str):
            return v
        elif isinstance(v, float):
            return "{:.2f}".format(v)
        elif isinstance(v, int):
            # fixme: may just return integer type
            return "{}".format(v)
        elif np.issubdtype(type(v), np.inexact):
            return "{:.2f}".format(v)
        elif np.issubdtype(type(v), np.integer):
            return "{}".format(v)
        else:
            raise NotImplementedError("Unsupported cell type for comparison, got: {}.".format(type(v)))

    def equal_tb(self, actual, expected):
        # logger.info("# calling equal_tb")
        # print("# expected type is: {}".format(expected.dtypes.tolist()))
        # print("# actual type is: {}".format(actual.dtypes.tolist()))
        # print("# actual is: {}".format(actual))
        # first convert tables into numpy arrays (dump the colnames)
        av = actual.values
        ev = expected.values
        # logger.info("# 0000 av:\n{}".format(av))
        # logger.info("# 0000 ev:\n{}".format(ev))
        # compare shape: this is necessary since some array may broadcast
        if av.shape != ev.shape:
            raise EqualityAssertion(tag="shape")

        # quick comparison
        res0 = av == ev
        if isinstance(res0, bool):
            if res0:
                # exact match, good to go
                return
            else:
                # numpy asserts that they are not equal for whatever reason
                raise EqualityAssertion(tag="numpy")

        # normalize the array for comparison
        av = self._normalize_cell_value(av)
        ev = self._normalize_cell_value(ev)
        # print("# av: {}".format(av))
        # print("# ev: {}".format(ev))
        # logger.info("# av:\n{}".format(av))
        # logger.info("# ev:\n{}".format(ev))

        # column-wise signature comparison
        # then store every column as a set and compare
        av_column_sets = [ frozenset(av[:,i].tolist()) for i in range(av.shape[1]) ]
        ev_column_sets = [ frozenset(ev[:,i].tolist()) for i in range(ev.shape[1]) ]
        # sort and compare
        av_column_sets = sorted(av_column_sets, key=lambda x: hash(x), reverse=False)
        ev_column_sets = sorted(ev_column_sets, key=lambda x: hash(x), reverse=False)
        res_column = all([av_column_sets[i]==ev_column_sets[i] for i in range(len(av_column_sets))])
        if not res_column:
            raise EqualityAssertion(tag="column")

        # row-wise signature comparison
        # then store every row as a set and compare
        av_row_sets = [ frozenset(av[i,:].tolist()) for i in range(av.shape[0]) ]
        ev_row_sets = [ frozenset(ev[i,:].tolist()) for i in range(ev.shape[0]) ]
        # sort and compare
        av_row_sets = sorted(av_row_sets, key=lambda x: hash(x), reverse=False)
        ev_row_sets = sorted(ev_row_sets, key=lambda x: hash(x), reverse=False)
        res_row = all([av_row_sets[i]==ev_row_sets[i] for i in range(len(av_row_sets))])
        if not res_row:
            raise EqualityAssertion(tag="row")

        # if you are here, you are good to go
        return

    # hijack the original eval method to add detection of context nslot
    def eval(self, prog: Node, inputs: List[Any]) -> Any:
        '''
        Interpret the Given AST in post-order. Assumes the existence of `eval_XXX` method where `XXX` is the name of a function defined in the DSL.
        '''
        class NodeVisitor(GenericVisitor):
            _interp: PostOrderInterpreter
            # fixme: legacy code with a different notion of context, try to remove this later
            # _context: Context

            def __init__(self, interp):
                self._interp = interp
                # fixme: legacy code with a different notion of context, try to remove this later
                # self._context = Context()

            # fixme: legacy code with a different notion of context, try to remove this later
            # def visit_with_context(self, node: Node):
            #     self._context.observe(node)
            #     res = self.visit(node)
            #     self._context.finish(node)
            #     return res

            # note: for atom node, to support parameter level conflict-driven learning, 
            #       use ??? to get the value, not the original eval_??? methods in Trinity
            #       and eventually return the node itself
            # note: in this version, every atom node is required to have tag and "cpos" field
            def visit_atom_node(self, atom_node: AtomNode):
                tmp_prod_id = atom_node.production.id
                # note: use self._interp to refer to the self in eval
                self._interp._current_combination = tuple([
                    tmp_prod_id if i==atom_node.tag["cpos"] else self._interp._current_combination[i] 
                    for i in range(len(self._interp._current_combination))
                ])
                return atom_node

            def visit_param_node(self, param_node: ParamNode):
                param_index = param_node.index
                if param_index >= len(inputs):
                    msg = 'Input parameter access({}) out of bound({})'.format(
                        param_index, len(inputs))
                    raise GeneralError(msg)
                return inputs[param_index]

            def visit_apply_node(self, apply_node: ApplyNode):
                # fixme: legacy code with a different notion of context, try to remove this later
                # in_values = [self.visit_with_context(x) for x in apply_node.args]
                # self._context.pop()
                in_values = [self.visit(x) for x in apply_node.args]
                method_name = self._eval_method_name(apply_node.name)
                method = getattr(self._interp, method_name,
                                 self._method_not_found)
                return method(apply_node, in_values)

            def _method_not_found(self, apply_node: ApplyNode, arg_values: List[Any]):
                msg = 'Cannot find required eval method: "{}"'.format(
                    self._eval_method_name(apply_node.name))
                raise NotImplementedError(msg)

            @staticmethod
            def _eval_method_name(name):
                return 'eval_' + name

        node_visitor = NodeVisitor(self)
        try:
            # try if this node is a root node ("skeleton" field only exists in root node)
            if prog.tag is not None:
                if "skeleton" in prog.tag:
                    # print("DEBUG: {}".format(prog.tag))
                    # yes it's root
                    # then initialize set the _current_combination
                    self._current_combination = tuple([None for _ in range(prog.tag["nslot"])])
            # fixme: legacy code with a different notion of context, try to remove this later
            # return node_visitor.visit_with_context(prog)
            return node_visitor.visit(prog)
        except InterpreterError as e:
            raise
            # fixme: legacy code with a different notion of context, try to remove this later
            # e.context = node_visitor._context
            # raise e from None

    def _get_context(self, comb, nodes):
        # this extracts the interpreter context (not the visitor context) from the interpreter combination
        return tuple([
            None if i in [p.tag["cpos"] for p in nodes] else comb[i]
            for i in range(len(comb))
        ])

    # ================================== #
    # ======== enum productions ======== #
    # ================================== #

    # fixme: merge with NodeVisitor later
    def _eval_method_name(self, name):
        return 'eval_' + name

    # main entrance of evaluating an atom node
    def _eval_atom_node(self, node):
        node_type = node.type.name
        method_name = self._eval_method_name(node_type)
        method = getattr(self, method_name)
        return method(node.data)

    # note: use this method in EnumAssertion
    def _eval_enum_prod(self, prod):
        prod_type = prod.lhs.name
        method_name = self._eval_method_name(prod_type)
        method = getattr(self, method_name)
        return method(prod.rhs[0])

    # can only be called by _eval_atom_node
    def eval_ColInt(self, v):
        return int(v)

    # can only be called by _eval_atom_node
    # def eval_SmallInt(self, v):
    #     return int(v)
    def eval_ConstVal(self, v):
        if v.endswith("@Float"):
            return float(v[:-6])
        elif v.endswith("@Int"):
            return int(v[:-4])
        elif v.endswith("@Str"):
            return v[:-4]
        else:
            raise ComponentError("ConstVal.")

    # can only be called by _eval_atom_node
    def eval_ColList(self, v):
        # question: is this true?
        return [int(p) for p in v]

    # can only be called by _eval_atom_node
    def eval_AggrFunc(self, v):
        return v

    # can only be called by _eval_atom_node
    def eval_NumFunc(self, v):
        return v

    # can only be called by _eval_atom_node
    def eval_BoolFunc(self, v):
        return v

    # ====================================== #
    # ======== function productions ======== #
    # ====================================== #

    # interpret collist into column ints
    # note-important: based on validate_collist, the list should either be all positive or all negative
    #                 otherwise this function won't work as expected
    # fixme: maybe add an assertion?
    def explain_collist(self, arg_ncol, arg_collist):
        # print("# explain: arg_ncol={}, arg_collist={}".format(arg_ncol, arg_collist))

        ret_collist = list(range(arg_ncol))
        if arg_collist[0] >= 0:
            # positive list
            ret_collist = [p for p in arg_collist]
        else:
            # negative list
            for p in arg_collist:
                if p == -99:
                    ret_collist.remove(0)
                else:
                    ret_collist.remove(-p)

        # print("# YES!")
        return ret_collist

    # this validates that the collist is not stupid
    # note: this should be called with in assertEnum, and before you call the explain function
    def validate_collist(self, arg_ncol, arg_collist):
        # print("# validate: arg_ncol={}, arg_collist={}".format(arg_ncol, arg_collist))
        # arg_collist is the original collist (before explanation)

        if len(arg_collist) != len(list(set(arg_collist))):
            # don't include duplicates
            # e.g., -1, -1, will cause ValueError in .remove(x) in explain_collist
            return False

        # note-important: don't mix positive and negative ints
        if max(arg_collist) >= 0 and min(arg_collist) < 0:
            return False

        for p in arg_collist:
            if p == 0:
                if -99 in arg_collist:
                    return False
                elif 0 >= arg_ncol:
                    return False
                else:
                    continue
            elif p == -99:
                if 0 in arg_collist:
                    return False
                if 0 >= arg_ncol:
                    return False
                else:
                    continue
            elif p > 0:
                if p >= arg_ncol:
                    return False
                elif -p in arg_collist:
                    return False
                else:
                    continue
            elif p < 0:
                if -p >= arg_ncol:
                    return False
                elif -p in arg_collist:
                    return False
                else:
                    continue
        # print("# YES~")
        return True

    # todo: add negative argument support
    # info: benchmarks/test/1
    def eval_select(self, node, args):
        arg_tb, node_collist = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="select:cached")
                else:
                    return self._cached_outputs[arg_sig]
        
        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_collist])
        self.assertEnum(node, _current_context, self._current_combination,
            # this makes sure the original colist is not stupid
            lambda comb: ( lambda x: self.validate_collist(arg_ncol, x) )(
                # original collist
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) ),
            ),
            tag="select:0",
        )

        # explain collist after the previous assertion holds
        arg_collist = self.explain_collist(arg_ncol, arg_collist)

        # note: the previous assertion already contains the following one
        # self.assertEnum(node, _current_context, self._current_combination,
        #     # note: to make sure the argument level pruning is working, pay attention to the following rules
        #     #       1. lambda function must refer to comb for getting production id, not the current node, so that
        #     #          the enumerator can re-use the lambda function
        #     # note: nested lambda to store temp variable
        #     lambda comb: (lambda x: max(x) < arg_ncol)(
        #         # note: now provide explained collist
        #         self.explain_collist(
        #             arg_ncol,
        #             self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) ),
        #         ),
        #     ),
        #     tag="select:1",
        # )

        try:
            tmp_cols = arg_tb.columns[arg_collist]
            ret_tb = arg_tb.loc[:, tmp_cols]
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="select")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb

    # info: benchmarks/test/2
    def eval_unite(self, node, args):
        arg_tb, node_col0, node_col1 = args
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="unite:cached")
                else:
                    return self._cached_outputs[arg_sig]

        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_col0, node_col1])
        self.assertEnum(node, _current_context, self._current_combination,
            # note: nested lambda to store temp variable
            lambda comb: (lambda x0,x1: x0 < arg_ncol and x1 < arg_ncol and x0 != x1)(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) )
            ),
            tag="unite:0",
        )

        try:
            tmp_col0 = arg_tb.columns[arg_col0]
            tmp_col1 = arg_tb.columns[arg_col1]
            tmp_colname = self.fresh_colname()
            ret_tb = arg_tb.assign(**{tmp_colname:arg_tb[tmp_col0].astype(str) + "_" + arg_tb[tmp_col1].astype(str)})
            # then remove the original columns
            ret_tb = ret_tb.drop([tmp_col0, tmp_col1], axis=1)
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="unite")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb

    # todo: add dynamic detection of delimiter
    # info: benchmarks/test/3
    def eval_separate(self, node, args):
        arg_tb, node_col = args
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="separate:cached")
                else:
                    return self._cached_outputs[arg_sig]

        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_col])
        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(node, _current_context, self._current_combination,
            # lambda comb: (lambda x: x < arg_ncol and pd.api.types.is_string_dtype(arg_tb[arg_tb.columns[x]]) )(
            # note: closure chooses dtype list to save memory?
            # note: nested lambda to store temp variable
            lambda comb: (lambda x: x < arg_ncol and pd.api.types.is_string_dtype(_current_dtypes[x]) )(
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) ),
            ),
            tag="separate:0",
        )

        try:
            tmp_col = arg_tb.columns[arg_col]
            # determine delimiter
            tmp_delimiter = None
            tmp_strtb = str(arg_tb)
            if "-" in tmp_strtb:
                # logger.info("# delimiter is -")
                tmp_delimiter = "-"
            elif "." in tmp_strtb:
                # logger.info("# delimiter is .")
                tmp_delimiter = "."
            else:
                # logger.info("# delimiter is None")
                tmp_delimiter = None
            ret_tb = pd.concat([
                arg_tb, 
                pd.DataFrame(arg_tb[tmp_col].str.split(tmp_delimiter).tolist(), columns=[self.fresh_colname(),self.fresh_colname()]) 
            ],axis=1)
            # then remove the original columns
            ret_tb = ret_tb.drop([tmp_col], axis=1)
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="separate")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb

    # info: benchmarks/test/4
    def eval_gather(self, node, args):
        arg_tb, node_collist = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="gather:cached")
                else:
                    return self._cached_outputs[arg_sig]

        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_collist])
        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(node, _current_context, self._current_combination,
            # x0: this makes sure the original colist is not stupid
            # self.explain_collist(x0): normal gather check
            # note-important: to ensure x0 comes before self.explain_collist(x0) checks, merge them into one assertEnum
            lambda comb: ( lambda x0: self.validate_collist(arg_ncol, x0) and \
                                      len(set([_current_dtypes[p] for p in self.explain_collist(arg_ncol, x0)]))==1 )(
                # original collist
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) ),
            ),
            tag="gather:0",
        )

        # explain collist after the previous assertion holds
        arg_collist = self.explain_collist(arg_ncol, arg_collist)

        try:
            tmp_cols = arg_tb.columns[arg_collist]
            tmp_rcols = [p for p in arg_tb.columns if p not in tmp_cols]
            ret_tb = pd.melt(
                arg_tb, 
                id_vars=tmp_rcols, 
                value_vars=tmp_cols, 
                var_name=self.fresh_colname(), 
                value_name=self.fresh_colname(),
            )
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="gather")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb

    # info: benchmarks/test/5
    def eval_spread(self, node, args):
        arg_tb, node_col0, node_col1 = args
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="spread:cached")
                else:
                    return self._cached_outputs[arg_sig]

        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_col0, node_col1])
        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(node, _current_context, self._current_combination,
            # note: nested lambda to store temp variable
            lambda comb: (lambda x0, x1: x0 < arg_ncol and x1 < arg_ncol and x0 != x1 and pd.api.types.is_string_dtype(_current_dtypes[x0]))(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) )
            ),
            tag="spread:0",
        )

        try:
            tmp_col0 = arg_tb.columns[arg_col0]
            tmp_col1 = arg_tb.columns[arg_col1]
            tmp_rcols = [p for p in arg_tb.columns if p not in [tmp_col0, tmp_col1]]
            ret_tb = pd.pivot(arg_tb, index=tmp_rcols, columns=[tmp_col0], values=[tmp_col1])
            # flatten multiple indices
            ret_tb = self.flatten_index(ret_tb)
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="spread")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb

    # info: benchmarks/test/6
    def eval_mutate(self, node, args):
        arg_tb, node_op, node_col0, node_col1 = args
        arg_op = self._eval_atom_node(node_op)
        arg_col0 = self._eval_atom_node(node_col0)
        arg_col1 = self._eval_atom_node(node_col1)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="mutate:cached")
                else:
                    return self._cached_outputs[arg_sig]

        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_op, node_col0, node_col1])
        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(node, _current_context, self._current_combination,
            # note: nested lambda to store temp variable
            lambda comb: (lambda x0, x1: x0 < arg_ncol and \
                                         x1 < arg_ncol and \
                                         x0 != x1 and \
                                         pd.api.types.is_numeric_dtype(_current_dtypes[x0]) and \
                                         pd.api.types.is_numeric_dtype(_current_dtypes[x1]))(
                self._eval_enum_prod( self._spec.get_production( comb[node_col0.tag["cpos"]] ) ),
                self._eval_enum_prod( self._spec.get_production( comb[node_col1.tag["cpos"]] ) )
            ),
            tag="mutate:0",
        )

        tmp_op = None
        if arg_op == "/":
            tmp_op = lambda x,y: x/y
        elif arg_op == "+":
            tmp_op = lambda x,y: x+y
        else:
            raise NotImplementedError("Unsupported NumFunc, got: {}.".format(arg_op))

        try:
            tmp_colname = self.fresh_colname()
            tmp_col0 = arg_tb.columns[arg_col0]
            tmp_col1 = arg_tb.columns[arg_col1]
            ret_tb = arg_tb.assign(**{tmp_colname:tmp_op(arg_tb[tmp_col0], arg_tb[tmp_col1])})
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="mutate")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb

    # info: benchmarks/test/7
    def eval_filter(self, node, args):
        arg_tb, node_op, node_col, node_int = args
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_int = self._eval_atom_node(node_int)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="filter:cached")
                else:
                    return self._cached_outputs[arg_sig]

        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_op, node_col, node_int])
        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(node, _current_context, self._current_combination,
            # note: nested lambda to store temp variable
            # lambda comb: (lambda x: x < arg_ncol and pd.api.types.is_numeric_dtype(_current_dtypes[x]) )(
            lambda comb: (lambda x: x < arg_ncol)(
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) )
            ),
            tag="filter:0",
        )

        tmp_op = None
        if arg_op == "==":
            tmp_op = lambda x,y: x==y
        elif arg_op == ">":
            tmp_op = lambda x,y: x>y
        elif arg_op == "<":
            tmp_op = lambda x,y: x<y
        elif arg_op == "!=":
            tmp_op = lambda x,y: x!=y
        else:
            raise NotImplementedError("Unsupported BoolFunc, got: {}.".format(arg_op))

        try:
            tmp_colname = self.fresh_colname()
            tmp_col = arg_tb.columns[arg_col]
            ret_tb = arg_tb[tmp_op(arg_tb[tmp_col],arg_int)]
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="filter")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb

    # info: benchmarks/test/8
    def eval_group(self, node, args):
        arg_tb, node_collist, node_op, node_col = args
        arg_collist = self._eval_atom_node(node_collist)
        arg_op = self._eval_atom_node(node_op)
        arg_col = self._eval_atom_node(node_col)
        arg_nrow, arg_ncol = arg_tb.shape

        # value caching
        if self._value_caching:
            # compute signature
            arg_sig = (str(node), self.hash_tb(arg_tb))
            if arg_sig in self._cached_outputs:
                if self._cached_outputs[arg_sig] is None:
                    # special token for ComponentError
                    raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="group:cached")
                else:
                    return self._cached_outputs[arg_sig]

        # extract the context from combination
        _current_context = self._get_context(self._current_combination, [node_collist, node_op, node_col])
        _current_dtypes = arg_tb.dtypes.tolist()
        self.assertEnum(node, _current_context, self._current_combination,
            # this makes sure the original colist is not stupid
            # note-important: to ensure x0 comes before self.explain_collist(x0)/y checks, merge them into one assertEnum
            lambda comb: ( lambda x0,y: self.validate_collist(arg_ncol, x0) and \
                                        y < arg_ncol and \
                                        pd.api.types.is_numeric_dtype(_current_dtypes[y]) )(
                # x0: original collist
                self._eval_enum_prod( self._spec.get_production( comb[node_collist.tag["cpos"]] ) ),
                # y
                self._eval_enum_prod( self._spec.get_production( comb[node_col.tag["cpos"]] ) ),
            ),
            tag="group:0",
        )

        # explain collist after the previous assertion holds
        arg_collist = self.explain_collist(arg_ncol, arg_collist)

        try:
            tmp_colname = self.fresh_colname()
            tmp_collist = [arg_tb.columns[p] for p in arg_collist]
            tmp_col = arg_tb.columns[arg_col]
            if arg_op == "min":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].min().to_frame()
            elif arg_op == "max":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].max().to_frame()
            elif arg_op == "sum":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].sum().to_frame()
            elif arg_op == "mean":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].mean().to_frame()
            elif arg_op == "count":
                ret_tb = arg_tb.groupby(by=tmp_collist)[tmp_col].count().to_frame()
            else:
                raise NotImplementedError("Unsupported AggrFunc, got: {}".format(arg_op))
            ret_tb = self.flatten_index(ret_tb)
            ret_tb = ret_tb.rename(columns={tmp_col:tmp_colname})
            # print("# ret_tb: {}".format(ret_tb))
        except:
            if self._suppress_pandas_exception:
                if self._value_caching:
                    self._cached_outputs[arg_sig] = None
                raise ComponentError(context=self._current_combination, condition=lambda comb: False, tag="group")
            else:
                raise
        if self._value_caching:
            self._cached_outputs[arg_sig] = ret_tb
        return ret_tb






