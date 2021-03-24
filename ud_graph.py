# Course: 
# Author: 
# Assignment: 
# Description:


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        TODO: Write this implementation
        """
        # SENTINEL = []
        # self.adj_list[v] = SENTINEL

        if v not in self.adj_list:
            self.adj_list[v] = []

    def add_edge(self, u, v):

        if u == v:
            return
        if u not in self.adj_list.keys():
            self.add_vertex(u)
        if v not in self.adj_list.keys():
            self.add_vertex(v)

        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)

        if u not in self.adj_list[v]:
            self.adj_list[v].append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """
        TODO: Write this implementation
        """
        if u not in self.adj_list.keys() or v not in self.adj_list.keys():
            return

        if (v, u) not in self.get_edges():
            return

        self.adj_list[v].remove(u)
        self.adj_list[u].remove(v)

    def remove_vertex(self, v: str) -> None:
        """
        TODO: Write this implementation
        """
        if v not in self.adj_list:
            return
        del self.adj_list[v]

        for i in self.adj_list:
            if v in self.adj_list[i]:
                self.adj_list[i].remove(v)

    def get_vertices(self) -> []:
        """
        TODO: Write this implementation
        """
        # vertices = list(self.adj_list.keys())
        # for key in self.adj_list.keys():
        #    vertices.append(key)
        return list(self.adj_list.keys())

    def remove_duplicates(self, list):

        output, seen = [], set()
        for item in list:
            t1 = tuple(item)
            if t1 not in seen and tuple(reversed(item)) not in seen:
                seen.add(t1)
                output.append(item)

        return output

    def get_edges(self) -> []:
        """
        TODO: Write this implementation
        """
        edges = []
        for k, v in self.adj_list.items():
            for i in v:
                edges.append((k, i))

        output = self.remove_duplicates(edges)

        return output

        # return edges

    def convert_string(self, string):
        list = []
        list[:0] = string
        return list

    def is_valid_path(self, path: []) -> bool:
        """
        TODO: Write this implementation
        """
        edges = []
        for k, v in self.adj_list.items():
            for i in v:
                edges.append((k, i))

        output = self.remove_duplicates(edges)

        vert = self.get_vertices()

        for i in range(len(path) - 1):
            adjacent_route = tuple(sorted(path[i] + path[i + 1]))
            if adjacent_route not in output and adjacent_route not in vert:
                return False

        if len(path) == 1:
            if path[0] not in vert:
                return False

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        if v_start not in self.get_vertices():
            return []
        if v_start == v_end:
            new_out = [v_start]
            return new_out

        visited = []
        out = []
        self._dfs(v_start, v_end, visited, out)

        new_out = []

        for i in out:
            new_out.append(i)
            if i == v_end:
                break
        return new_out


    def _dfs(self, v_start, v_end=None, visited=[], ret=[]):

        visited.append(v_start)
        # if v_end in visited:
        # if v_start == v_end:
        #     return ("end", v_end, visited, visited)
        ret.append(v_start)
        stack = self.adj_list[v_start]
        stack.sort()
        for itm in stack:
            if itm not in visited:
                self._dfs(itm, v_end, visited, ret)

    def bfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        if v_start not in self.get_vertices():
            return []
        visited = []
        keys = []
        for itm in self.get_vertices():
            keys.append(itm)
            visited.append(False)
        ret = []
        stack = []
        stack.append(v_start)
        visited[keys.index(v_start)] = True
        while stack:
            v_start = stack.pop(0)
            ###
            ret.append(keys.index(v_start))
            if v_start == v_end:
                break
            to_append = []
            for i in self.adj_list[v_start]:
                if visited[keys.index(i)] == False:
                    to_append.append(i)
                    visited[keys.index(i)] = True
            to_append.sort()
            for itm in to_append:
                stack.append(itm)
        ret2 = []
        ctr = 0
        while ctr < ret.__len__():
            ret2.append(keys[ret[ctr]])
            ctr = ctr + 1
        return ret2

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for vertex in self.adj_list[v]:
            if visited[vertex] == False:
                # Update the list
                temp = self.DFSUtil(temp, vertex, visited)
        return temp


    def count_connected_components(self):
        """
        TODO: Write this implementation
        """
        visited = {node: False for node in self.adj_list}
        connected_sections = []
        for vertex in self.adj_list:
            if not visited[vertex]:
                temp = []
                connected_sections.append(self.DFSUtil(temp, vertex, visited))
        return len(connected_sections)

    def has_cycle(self):
        """
        redo make clean
        :return:
        """
        visited = {node: False for node in self.adj_list}
        found_cycle = [False]
        for node in self.adj_list:  # - Visit all nodes.
            if not visited[node]:
                self.dfs_visit(self.adj_list, node, found_cycle, node, visited)  # - u is its own predecessor initially
            if found_cycle[0]:
                break
        return found_cycle[0]

    def dfs_visit(self, adj_list, current_node, found_cycle, previous, visited):
        """
        helper function for has_cycle. Determines if there is a cycle using dfs.
        :param adj_list: an adjacency list
        :param current_node: the current node in the adjacency list
        :param found_cycle: Whether a cycle has been found
        :param previous: the previous node
        :param visited: list of whether the nodes have been visited or not
        :return: Boolean of if the cycle has been found
        """
        if found_cycle[0]:  # - Stop dfs if cycle is found.
            return
        # mark visited as true
        visited[current_node] = True
        # iterate through the neighbors
        for neighbor in adj_list[current_node]:
            # if the neighbor node is visited and not the previous node
            if visited[neighbor] and neighbor != previous:
                # cycle exists
                found_cycle[0] = True
                return
            if not visited[neighbor]:
                self.dfs_visit(adj_list, neighbor, found_cycle, current_node, visited)


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)

    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    print(g)
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)
    print(g)

    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g)
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')

    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())


