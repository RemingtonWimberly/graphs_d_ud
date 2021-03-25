# Course: CS261 - Data Structures
# Author: 
# Assignment: 
# Description: 

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
        TODO: Write this implementation
        """
        self.v_count += 1
        self.adj_matrix.append([0] * self.v_count)

        for i in range(self.v_count - 1):
            self.adj_matrix[i].append(0)

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        TODO: Write this implementation
        """
        if src >= self.v_count or dst >= self.v_count or weight <= 0 or src == dst:
            # if src >= self.v_count or dst >= self.v_count or weight <= 0 or src == dst:
            return
        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        TODO: Write this implementation
        """
        if dst >= self.v_count or src >= self.v_count:
            return
        if (src, dst) not in self.get_new_edges() and (dst,src) not in self.get_new_edges():
            return
        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """
        TODO: Write this implementation
        """

        verticies = []
        for i in list(range(len(self.adj_matrix))):
            verticies.append(i)

        return verticies

    def get_string_vertices(self) -> []:
        """
        TODO: Write this implementation
        """
        verticies = []
        for i in list(range(len(self.adj_matrix))):
            verticies.append(str(i))

        return verticies

    def get_v_string(self):
        meow = list(range(len(self.adj_matrix)))
        meow2 = []
        for i in meow:
            meow2.append('{}'.format(i))

    def get_edges(self) -> []:
        """
        TODO: Write this implementation
        """
        edge_list = []
        edges = self._get_edges(self.adj_matrix)
        for i in edges:
            edge_list.append(i)

        return edge_list

    def get_new_edges(self) -> []:
        """
        TODO: Write this implementation
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
        TODO: Write this implementation
        """
        for i in range(len(path) - 1):
            if self.adj_matrix[path[i]][path[i + 1]] == 0:
                return False

        return True

    def convert_matrix_to_Adj_list(self, matrix):

        l = matrix

        graph = defaultdict(list)
        edges = set()

        for i, v in enumerate(l, 0):
            for j, u in enumerate(v, 0):
                if u != 0 and frozenset([i, j]) not in edges:
                    edges.add(frozenset([i, j]))
                    # graph[i].append({j: u})
                    graph[i].append(j)
                    # graph[i].append(u)
        return graph

    # def dfs(self, v_start, v_end=None) -> []:
    #     """
    #     TODO: Write this implementation
    #     """
    #     visited = []
    #     out = []
    #     self._dfs(v_start, v_end, visited, out)
    #
    #     return out
    #     # use range
    #     # for u in range(foo, v_end):

    def dfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        if str(v_start) not in self.get_string_vertices():
            return []
        if v_start == v_end:
            new_out = [v_start]
            return new_out

        visited = []
        out = []
        adj_list = dict(self.convert_matrix_to_Adj_list(self.adj_matrix))
        self._dfs(adj_list, v_start, v_end, visited, out)

        new_out = []

        for i in out:
            new_out.append(i)
            if i == v_end:
                break
        return new_out

    def _dfs(self,adj_list, v_start, v_end=None, visited=[], ret=[]):

        visited.append(v_start)
        ret.append(v_start)
        if not adj_list[v_start]:
            return [v_start]
        stack = adj_list[v_start]
        stack.sort()
        for itm in stack:
            if itm not in visited:
                self._dfs(adj_list, itm, v_end, visited, ret)

    def bfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        adj_list = dict(self.convert_matrix_to_Adj_list(self.adj_matrix))
        visited = []
        keys = []
        for itm in self.get_string_vertices():
            keys.append(itm)
            visited.append(False)
        ret = []
        stack = []
        stack.append(v_start)
        visited[keys.index('{}'.format(v_start))] = True
        while stack:
            v_start = stack.pop(0)
            ###
            ret.append(keys.index('{}'.format(v_start)))
            if v_start == v_end:
                break
            to_append = []
            for i in adj_list[v_start]:
                if visited[keys.index('{}'.format(i))] == False:
                    to_append.append(i)
                    visited[keys.index('{}'.format(i))] = True
            to_append.sort()
            for itm in to_append:
                stack.append(itm)
        ret2 = []
        ctr = 0
        while ctr < ret.__len__():
            ret2.append(keys[ret[ctr]])
            ctr = ctr + 1
        ret3 = []
        for i in ret2:
            ret3.append(int(i))
        return ret3

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
        visited = [False] * len(self.get_string_vertices())
        stack = [False] * len(self.get_string_vertices())
        for node in range(len(self.get_string_vertices())):
            if not visited[node]:
                if self._has_cycle(node, visited, stack):
                    return True
        return False

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
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    practice_edges = [()]

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)

    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_string_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_string_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)

    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
