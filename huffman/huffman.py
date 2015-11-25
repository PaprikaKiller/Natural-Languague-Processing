from itertools import groupby
from heapq import *

from .node import Node


def huffman(input_huffman):

    item_queue = [Node(a, len(list(b))) for a, b in groupby(sorted(input_huffman))]
    heapify(item_queue)

    while len(item_queue) > 1:
        l = heappop(item_queue)
        r = heappop(item_queue)
        n = Node(None, r.weight+l.weight)
        n.set_children(l,r)
        heappush(item_queue, n)

    codes = {}

    def code_it(s, node):
        if node.item:
            if not s:
                codes[node.item] = "0"
            else:
                codes[node.item] = s
        else:
            code_it(s+"0", node.left)
            code_it(s+"1", node.right)

    code_it("", item_queue[0])

    return codes, "".join([codes[a] for a in input_huffman])
