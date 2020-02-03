"""
ComptoxAI graphs.

A typical graph workflow looks something like the following:

>>> from comptox_ai import Graph
>>> G = Graph.from_neo4j(config_file = "./CONFIG.cfg")
>>> G.convert_inplace(to='networkx')
>>> A = G.get_adjacency()
>>> GS = G.convert(to='graphsage')
"""

import numpy as np
import scipy.sparse
import neo4j
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
from queue import Queue

from abc import abstractmethod
from typing import List, Iterable, Union
import os
import json
from textwrap import dedent

import ipdb
from tqdm import tqdm

from comptox_ai.cypher import queries
from comptox_ai.utils import execute_cypher_transaction
from comptox_ai.graph.metrics import vertex_count, ensure_nx_available
from .vertex import Vertex
from .subgraph import Subgraph
from .utils import Spinner

from .io import GraphDataMixin, Neo4j, NetworkX, GraphSAGE

class Graph(object):
    """
    Mixin class defining the standard interface for all graph data structures
    used in ComptoxAI.
    
    Classes implementing this interface will have a minimal set of data
    accession and manipulation routines with identical type signatures, each of
    which is inspired by the NetworkX API (but modified to more fluidly handle
    heterogeneous graphs).

    Parameters
        ----------
        node_map : dict, default=None
            Map of neo4j node ids to consecutive integers.
        edge_map : dict, default=None
            Map of neo4j edge ids to consecutive integers.
        node_classes : list of str, default=None
            List of ontology classes present in the set of graph nodes. If 
            None, the graph will be parsed as a homogeneous graph (i.e., all
            nodes share the same feature space).
        edge_classes : list of str, default=None
            List of ontology object properties present in the set of graph
            edges (called "edge labels" by Neo4j). If None, no semantic
            information will be bound to edges in the graph.
        node_features : {array-like, dict of array-like}, default=None
            One or more array-like data structures containing node features.
            Must be compatible with node_classes - if node_classes is None,
            node_features should be either None or a single array-like. If
            node_classes is a list of length n, node_features should be either
            None or a dict of length n mapping each element of node_classes to
            its corresponding array-like of node features.
        edge_features : {array-like, dict of array-like}, default=None
            One or more array-like data structures containing edge features.
            Must be compatible with edge_classes - if edge_classes is None,
            edge_features should be either None or a single array-like. If
            edge_classes is a list of length m, edge_features should be either
            None or a dict of length m mapping each element of edge_classes to
            its corresponding array-like of edge features.
    """

    def __init__(self, data: GraphDataMixin):
        self._data = data

        if isinstance(data, Neo4j):
            self.format = 'neo4j'
        elif isinstance(data, NetworkX):
            self.format = 'networkx'
        elif isinstance(data, GraphSAGE):
            self.format = 'graphsage'

    def __repr__(self):
        return dedent(
            """
            ComptoxAI Graph
            ---------------
            Format:     {0}
            Node count: {1}
            Edge count: {2}
            """
        ).format(
            self.format,
            len(self.get_nodes()),
            len(self.get_edges())
        )

    def get_nodes(self):
        return self._data.nodes

    @abstractmethod
    def add_node(self, nodes: Union[List[tuple], tuple]):
        pass

    def get_edges(self):
        return self._data.edges

    @abstractmethod
    def add_edge(self, edges):
        pass

    @property
    @abstractmethod
    def id_map(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @property
    def is_heterogeneous(self):
        return self._data._is_heterogeneous

    @classmethod
    def from_neo4j(cls):
        raise NotImplementedError

    @classmethod
    def from_networkx(cls):
        raise NotImplementedError

    @classmethod
    def from_graphsage(cls, prefix: str, directory: str=None):
        """
        Create a new GraphSAGE data structure from files formatted according to
        the examples given in https://github.com/williamleif/GraphSAGE.

        The parameters should point to files with the following structure:

        {prefix}-G.json
            JSON file containing a NetworkX 'node link' instance of the input
            graph. GraphSAGE usually expects there to be 'val' and 'test'
            attributes on each node indicating if they are part of the
            validation and test sets, but this isn't enforced by ComptoxAI (at
            least not currently).

        {prefix}-id_map.json
            A JSON object that maps graph node ids (integers) to consecutive
            integers (0-indexed).

        {prefix}-class_map.json (OPTIONAL)
            A JSON object that maps graph node ids (integers) to a one-hot list
            of binary class membership (e.g., {2: [0, 0, 1, 0, 1]} means that
            node 2 is a member of classes 3 and 5). NOTE: While this is shown
            as a mandatory component of a dataset in GraphSAGE's documentation,
            we don't enforce that. NOTE: The notion of a class in terms of
            GraphSAGE is different than the notion of a class in heterogeneous
            network theory. Here, a 'class' is a label to be used in a
            supervised learning setting (such as classifying chemicals as
            likely carcinogens versus likely non-carcinogens).

        {prefix}-feats.npy (OPTIONAL)
            A NumPy ndarray containing numerical node features. NOTE: This
            serialization is currently not compatible with heterogeneous
            graphs, as GraphSAGE was originally implemented for
            nonheterogeneous graphs only.

        {prefix}-walks.txt (OPTIONAL)
            A text file containing precomputed random walks along the graph.
            Each line is a pair of node integers (e.g., the second fields in
            the id_map file) indicating an edge included in random walks. The
            lines should be arranged in ascending order, starting with the 
            first item in each pair.

        Parameters
        ----------
        prefix : str
            The prefix used at the beginning of each file name (see above for
            format specification).
        directory : str, default=None
            The directory (fully specified or relative) containing the data
            files to load.
        """

        nx_json_file = os.path.join(directory, "".join([prefix, '-G.json']))
        id_map_file = os.path.join(directory, "".join([prefix, '-id_map.json']))
        class_map_file = os.path.join(directory, "".join([prefix, '-class_map.json']))
        feats_map_file = os.path.join(directory, "".join([prefix, '-feats.npy']))
        walks_file = os.path.join(directory, "".join([prefix, '-walks.txt']))

        G = json_graph.node_link_graph(json.load(open(nx_json_file, 'r')))
        id_map = json.load(open(id_map_file, 'r'))

        try:
            class_map = json.load(open(class_map_file, 'r'))
        except FileNotFoundError:
            class_map = None

        try:
            feats_map = np.load(feats_map_file)
        except FileNotFoundError:
            feats_map = None

        try:
            walks = []
            with open(walks_file, 'r') as fp:
                for l in fp:
                    walks.append(l)
        except FileNotFoundError:
            walks = None

        graph_data = GraphSAGE(graph=G, node_map=id_map,
                               node_classes=class_map,
                               node_features=feats_map)

        return cls(data = graph_data)
    
    @classmethod
    def from_dgl(cls):
        raise NotImplementedError