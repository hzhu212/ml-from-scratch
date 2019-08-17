import numpy as np


class Node(object):
    """Node of compute graph"""
    def __init__(self, x, *args, **kw):
        if not isinstance(x, Node):
            raise ValueError('the input should be a compute graph Node object')
        x.next = self
        self.next = None
        self.grad = None
        self.init(*args, **kw)

    def init(self, *args, **kw):
        pass

    def fun(self, x):
        """节点中保存的基本函数"""
        pass

    def fun_grad(self, x, out):
        """基本函数的导函数"""
        pass

    def forward(self, x):
        """计算输出，同时缓存梯度"""
        out = self.fun(x)
        self.grad = self.fun_grad(x, out)
        return out

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return '<"{}" node of compute graph>'.format(str(self))


class ComputeGraph(object):
    """Compute Graph"""
    def __init__(self, inp, out):
        self.head = inp
        self.tail = out
        self.grad = None

    def forward(self, x):
        if self.head is None:
            raise ValueError('the graph is empty')
        out = x
        grad = 1.0
        node = self.head
        while node:
            out = node.forward(out)
            grad *= node.grad
            node = node.next
        self.grad = grad
        return out

    def __str__(self):
        node = self.head
        desc = []
        while node:
            desc.append(str(node))
            node = node.next
        return ' --> '.join(desc)


class Input(Node):
    """Input Node"""
    def __init__(self):
        self.next = None

    def fun(self, x):
        return x

    def fun_grad(self, x, out):
        return 1


class power(Node):
    """Node of power function"""
    def init(self, p):
        self.p = p

    def fun(self, x):
        return np.power(x, self.p)

    def fun_grad(self, x, out):
        return self.p * np.power(x, self.p - 1)

    def __str__(self):
        return '{}(., {})'.format(self.__class__.__name__, self.p)


class exp(Node):
    """Node of exp function"""
    def fun(self, x):
        return np.exp(x)

    def fun_grad(self, x, out):
        return out


class sin(Node):
    """Node of sin function"""
    def fun(self, x):
        return np.sin(x)

    def fun_grad(self, x, out):
        return np.cos(x)


inp = Input()
out = power(inp, 2)
print(repr(out))
out = exp(out)
out = sin(out)
graph = ComputeGraph(inp, out)
print(graph)


def f(x):
    return np.sin(np.exp(np.power(x, 2)))

def f_prime(x):
    t = np.exp(np.power(x, 2))
    return 2 * x * np.cos(t) * t


x = np.linspace(0, 1, 5)
out = graph.forward(x)
print('output: {!r}\ngradient: {!r}'.format(out, graph.grad))
print('verify output: {!r}\nverify gradient: {!r}'.format(f(x), f_prime(x)))

print('verify output pass:', np.all(f(x) == graph.forward(x)))
print('verify gradient pass:', np.all(f_prime(x) == graph.grad))
