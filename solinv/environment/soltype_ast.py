from copy import deepcopy
import igraph as ig

def insert_padding_node(arg_igraph, arg_e2n, arg_e2r, arg_root_id, arg_padding_token="<PAD>"):
    """Inserts an extra node at the very beginning (index=0) with padding token as label"""
    # assertions come first
    for i in range(len(arg_igraph.vs)):
        assert arg_igraph.vs[i].index == i, "Indices mismatch for node at {}, expected {} but got {}.".format(i, i, arg_igraph.vs[i].index)
    for i in range(len(arg_igraph.es)):
        assert arg_igraph.es[i].index == i, "Indices mismatch for edge at {}, expected {} but got {}.".format(i, i, arg_igraph.es[i].index)
    # first export nodes as records
    node_attributes = arg_igraph.vs["token"]
    edge_attributes = arg_igraph.es["token"]
    edge_tuples = [ arg_igraph.es[i].tuple for i in range(len(arg_igraph.es)) ]
    # inset one extra padding node at the beginning (index=0)
    node_attributes = [arg_padding_token] + node_attributes
    # shift all the indices in edge info by +1 to the id
    edge_tuples = [ 
        (edge_tuples[i][0]+1, edge_tuples[i][1]+1)
        for i in range(len(edge_tuples))
    ]
    new_igraph = ig.Graph(
        directed=True,
        n=len(node_attributes),
        vertex_attrs={"token": node_attributes},
        edges=edge_tuples,
        edge_attrs={"token": edge_attributes},
    )

    # process/shift the remaining components
    new_e2n = { dkey:arg_e2n[dkey]+1 for dkey in arg_e2n.keys() }
    new_e2r = {
        [ arg_e2r[dkey][i]+1 for i in range(len(arg_e2r[dkey])) ]
        for dkey in arg_e2r.keys()
    }
    new_root_id = arg_root_id + 1
    return new_igraph, new_e2n, new_e2r, new_root_id


def extract_tokens(d, f):
    """Extract tokens from an JSON AST, using extraction function f"""

    def inner(d):
        if type(d) not in [dict, list]:
            return set()
        elif type(d) == list:
            res = [inner(x) for x in d]
            return set().union(*res)
        else:
            res = [inner(v) for v in d.values()]
            return set(f(d)).union(*res)

    return inner(d)

def linearize(ast):
    """Turn lists into linked lists in an AST"""
    if type(ast) != dict:
        return ast
    worklist = [(k, v) for k, v in ast.items() if type(v) == list and len(v) > 0]
    for k, v in worklist:
        subtrees = list(map(linearize, ast[k]))
        # Make new tokens for head and next pointers
        # header pointer token = key to list + _first
        # next pointer token = key to list + _next
        # Note: In the future, we might want to change them to _first and _next (without the key prefix)
        ast[k + "_first"] = subtrees[0]
        try:
            for i in range(len(subtrees) - 1):
                subtrees[i][k + "_next"] = subtrees[i + 1]
        except TypeError:
            print(subtrees)
    # clean up
    for k, _ in worklist:
        del ast[k]
    return ast

def make_contract_ast(l_dfun):
    """Make an contract from a list of function declarations"""
    return {"tag": "<CONTRACT>", "DFun": l_dfun}

def preprocess_DFun_args(dfun):
    """Preprocess function arguments in function declaration `dfun`"""
    assert "DFun_args" in dfun
    dfun["DFun_args"] = [
        {"tag": "DFun_arg", "DFun_arg_name": name, "DFun_arg_type": t}
        for name, t in dfun["DFun_args"]
    ]
    return dfun

def label_vertices(ast, vi, vertices, var_v):
    """Label each node in the AST with a unique vertex id
    vi : vertex id counter
    vertices : list of all vertices (modified in place)
    """

    def inner(ast):
        nonlocal vi
        if type(ast) != dict:
            if type(ast) == list:
                # print(vi)
                pass
            return ast
        ast["vertex_id"] = vi
        vertices.append(ast["tag"])
        # if not (ast['tag'] in ['EVar', 'LvVar'] and ast['contents'] in var_v):
        vi += 1
        for k, v in ast.items():
            if k != "tag":
                inner(v)
        return ast

    return inner(ast)

def label_edges(ast, ei, edges, var_v):
    """Label each edge in the AST with a unique edge id
    ei : edge id counter
    edges : list of all edges (modified in place)
    """

    def inner(ast, p=None, edge_token=None):
        nonlocal ei
        if type(ast) != dict:
            if type(ast) == list:
                # print(ei)
                pass
            return ast
        # if this is a storage variable, connect to it directly
        if ast["tag"] == "<VAR>" and ast["Var_name"] in var_v:
            vi = var_v[ast["Var_name"]]
        else:
            vi = ast["vertex_id"]
        if p is not None:
            edges.append(((p, vi), edge_token))
            ei += 1
        # recurse
        for k, v in ast.items():
            if k != "tag":
                inner(v, vi, k)
        return ast

    return inner(ast)

def get_soltype_ast(contract_json):
    contract_name, contents = contract_json
    find = lambda l, tag: [x for x in l if type(x) == dict and x["tag"] == tag]
    # constructors
    l_dctor = find(contents, "DCtor")[0]
    # storage variables
    l_dvar = find(contents, "DVar")
    # functions
    l_dfun = find(contents, "DFun")

    # preprocess function arguments
    l_dfun2 = [preprocess_DFun_args(dfun) for dfun in l_dfun]
    contract = make_contract_ast(l_dfun2)

    # turn lists into linked lists
    contract2 = linearize(contract)

    # sanity check: vertex tokens and edge tokens
    v_tokens = sorted(
        list(
            extract_tokens(contract2, lambda d: [v for k, v in d.items() if k == "tag"])
        )
    )
    e_tokens = sorted(
        list(
            extract_tokens(contract2, lambda d: [k for k, v in d.items() if k != "tag"])
        )
    )
    # TODO: assert v_tokens \subset self.reserved_vertex_token_list
    # TODO: assert e_tokens \subset self.reserved_edge_token_list

    def inverse(d):
        """Return the inverse dictionary of d"""
        return {v: k for k, v in d.items()}

    # vertex index |-> storage variable name
    v_var = {i: v["DVar_name"] for i, v in enumerate(l_dvar)}
    # storage variable name |-> vertex index
    var_v = inverse(v_var)

    # reserve the first several vertices for storage variables
    vertices = sorted(["<VAR>" for _ in var_v])
    # print(vertices)
    # populate the vertex list
    contract3 = label_vertices(deepcopy(contract2), len(vertices), vertices, var_v)

    # populate the edge list
    edges = list()
    contract4 = label_edges(deepcopy(contract3), 0, edges, var_v)

    return contract4, vertices, edges, var_v

def soltype_ast_to_igraph(contract_ast, vs, es, var_v):
    g = ig.Graph(
        directed=True,
        n=len(vs),
        vertex_attrs={"token": vs},
        edges=[e for e, _ in es],
        edge_attrs={"token": [tk for _, tk in es]},
    )

    g.delete_vertices([v for v in g.vs if v.degree() == 0])

    root_id = contract_ast["vertex_id"]

    return var_v, {}, g, root_id
