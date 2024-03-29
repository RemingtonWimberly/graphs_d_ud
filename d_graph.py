# Course: CS261 - Data Structures
# Author: Remington Wimberly

from collections import defaultdict
from pprint import pprint
import sys
from heapq import heappop, heappush


class Node:
    def __init__(self, vertex, weight):
        self.vertex = vertex
        self.weight = weight

    # Override the `__lt__()` function to make `Node` class work with a min-heap
    def __lt__(self, other):
        return self.weight < other.weight

class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        adds a vertex to the graph
        """
        self.v_count += 1
        self.adj_matrix.append([0] * self.v_count)

        for i in range(self.v_count - 1):
            self.adj_matrix[i].append(0)

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        adds an edge to the graph
        """
        if src >= self.v_count or dst >= self.v_count or weight <= 0 or src == dst:
            # if src >= self.v_count or dst >= self.v_count or weight <= 0 or src == dst:
            return
        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes an edge from the graph
        """
        if dst >= self.v_count or src >= self.v_count:
            return
        if (src, dst) not in self.get_new_edges() and (dst,src) not in self.get_new_edges():
            return
        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """
        returns the verticies
        """

        verticies = []
        for i in list(range(len(self.adj_matrix))):
            verticies.append(i)

        return verticies

    def get_string_vertices(self) -> []:
        """
        returns the verticies as strings
        """
        verticies = []
        for i in list(range(len(self.adj_matrix))):
            verticies.append(str(i))

        return verticies

    def get_edges(self) -> []:
        """
        funciton to get the edges with the weights
        """
        edge_list = []
        edges = self._get_edges(self.adj_matrix)
        for i in edges:
            edge_list.append(i)

        return edge_list

    def get_new_edges(self) -> []:
        """
        function to get the edge without the weight
        """
        edge_list = []
        edges = self._get_edges(self.adj_matrix)
        for i in edges:
            edge_list.append(i[0:2])

        return edge_list

    def _get_edges(self, adj):

        """
        helper function for get_edges
        :param adj: adjacency matrix
        :return: generator object
        """

        for row, neighbors in enumerate(adj):
            for column, value in enumerate(neighbors):
                if value:
                    yield row, column, value

    def is_valid_path(self, path: []) -> bool:
        """
        determines if a path is valid
        """
        for i in range(len(path) - 1):
            if self.adj_matrix[path[i]][path[i + 1]] == 0:
                return False

        return True

    def convert_matrix_to_Adj_list(self, matrix):

        """ Converts a matrix to an adjacency list"""
        """ code in part borrowed from stack overflow"""

        adj_matrix = matrix

        graph = defaultdict(list)
        edges = set()

        for i, v in enumerate(adj_matrix, 0):
            for j, u in enumerate(v, 0):
                if u != 0 and frozenset([i, j]) not in edges:
                    edges.add(frozenset([i, j]))
                    graph[i].append(j)
        return graph

    def dfs(self, v_start, v_end=None) -> []:
        """
        depth first search
        """
        if str(v_start) not in self.get_string_vertices():
            return []
        if v_start == v_end:
            new_out = [v_start]
            return new_out

        visited = [False] * self.v_count
        out = []
        # adj_list = dict(self.convert_matrix_to_Adj_list(self.adj_matrix))
        # self._dfs(adj_list, v_start, v_end, visited, out)
        self._dfs(v_start, visited, out)

        new_out = []

        for i in out:
            new_out.append(i)
            if i == v_end:
                break
        return new_out

    # def _dfs(self,adj_list, v_start, v_end=None, visited=[], ret=[]):
    #     """ DFS helper funciton """
    #     visited.append(v_start)
    #     ret.append(v_start)
    #     if not adj_list[v_start]:
    #         return [v_start]
    #     stack = adj_list[v_start]
    #     stack.sort()
    #     for itm in stack:
    #         if itm not in visited:
    #             self._dfs(adj_list, itm, v_end, visited, ret)

    def _dfs(self, start, visited, ret):

        """ DFS helper function """
        # append current node
        ret.append(start)

        visited[start] = True

        for vert in range(self.v_count):

            if (not visited[vert]) and self.adj_matrix[start][vert] > 0:
                self._dfs(vert, visited, ret)



    def bfs(self, v_start, v_end=None) -> []:
        """ breadth first search """

        out = self._bfs(v_start)

        new_out = []

        for i in out:
            new_out.append(i)
            if i == v_end:
                break
        return new_out

    def _bfs(self, start) -> []:
        """ bfs helper function """

        return_list = [start]
        queue = [start]
        visited = [False] * len(self.adj_matrix)
        visited[start] = True
        while len(queue) > 0:
            vertex = queue.pop(0)

            for i in range(len(self.adj_matrix[vertex])):
                if self.adj_matrix[vertex][i] and not visited[i]:
                    return_list.append(i)
                    visited[i] = True
                    queue.append(i)

        return return_list


    def _has_cycle(self, v, visited, stack):
        """
        has cycle helper function
        """
        adj_list = self.convert_matrix_to_Adj_list(self.adj_matrix)
        visited[v] = True
        stack[v] = True

        for neighbour in adj_list[v]:
            if not visited[neighbour]:
                if self._has_cycle(neighbour, visited, stack):
                    return True
            elif stack[neighbour]:
                return True
        stack[v] = False
        return False

    def has_cycle(self):
        """
        This method returns True if there is at least one cycle in the graph. If the graph is acyclic,
        the method returns False.
        """
        # visited = [False] * self.v_count
        # stack = [False] * self.v_count
        # for node in range(self.v_count):
        #     if not visited[node]:
        #         if self._has_cycle(node, visited, stack):
        #             return True
        # return False
        adj_list = self.convert_matrix_to_Adj_list(self.adj_matrix)
        path = set()
        visited = set()

        def visit_nodes(node):
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for neighbour in adj_list.get(node, ()):
                if neighbour in path or visit_nodes(neighbour):
                    return True
            path.remove(node)
            return False

        return any(visit_nodes(v) for v in adj_list)

    def dijkstra(self, start):
        """

        This method implements the Dijkstra algorithm to compute the length of the shortest path
        from a given vertex to all other vertices in the graph. It returns a list with one value per
        each vertex in the graph, where value at index 0 is the length of the shortest path from
        vertex SRC to vertex 0, value at index 1 is the length of the shortest path from vertex SRC
        to vertex 1 etc. If the node is not reachable from source, returned value is infinity

        :param start: Starting node
        :return: an array of shortest distances from start node to every other node
        """

        distances_list = [float("inf")] * len(self.get_string_vertices())
        # set distance from start node to start node to zero
        distances_list[start] = 0

        visited_list = [False] * len(self.get_string_vertices())


        # while all node have not been visited
        while True:

            shortest_distance = float("inf")
            shortest_path_index = -1
            # get node with shortest distance from the start node
            for vertices in range(len(self.adj_matrix)):
                if distances_list[vertices] < shortest_distance and not visited_list[vertices]:
                    shortest_distance = distances_list[vertices]
                    # updates the shortest index to the node that has a path distance is less than inf and is not visited
                    shortest_path_index = vertices


            if shortest_path_index == -1:
                # all nodes have been visited and the shortest index is -1, dist is inf
                return distances_list

            # neighboring nodes not visited
            for vertices in range(len(self.adj_matrix[shortest_path_index])):
                # if the edge path is shorter update shortest path
                if distances_list[vertices] > distances_list[shortest_path_index] + self.adj_matrix[shortest_path_index][vertices] and self.adj_matrix[shortest_path_index][vertices] != 0:
                    distances_list[vertices] = distances_list[shortest_path_index] + self.adj_matrix[shortest_path_index][vertices]

            # mark visited as true
            visited_list[shortest_path_index] = True




if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    adj_list = DirectedGraph()
    print(adj_list)
    for _ in range(5):
        adj_list.add_vertex()
    print(adj_list)

    practice_edges = [()]

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        adj_list.add_edge(src, dst, weight)
    print(adj_list)

    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    adj_list = DirectedGraph()
    print(adj_list.get_edges(), adj_list.get_string_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    adj_list = DirectedGraph(edges)
    print(adj_list.get_edges(), adj_list.get_string_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    adj_list = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, adj_list.is_valid_path(path))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    adj_list = DirectedGraph(edges)
    for v_start in range(5):
        print(f'{v_start} DFS:{adj_list.dfs(v_start)} BFS:{adj_list.bfs(v_start)}')

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    adj_list = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        adj_list.remove_edge(src, dst)
        print(adj_list.get_edges(), adj_list.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        adj_list.add_edge(src, dst)
        print(adj_list.get_edges(), adj_list.has_cycle(), sep='\n')
    print('\n', adj_list)

    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    adj_list = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {adj_list.dijkstra(i)}')
    adj_list.remove_edge(4, 3)
    print('\n', adj_list)
    for i in range(5):
        print(f'DIJKSTRA {i} {adj_list.dijkstra(i)}')
