
# test for inherit
class A():
    def fun1(self):
        return self.fun2()

class B(A):
    def fun2(self):
        return "AAAA"

# right
test = B()
print(test.fun1())
# wrong
test2 = A()
print(test2.fun1())






# test for yield
# def fab(max):
#     n, a, b = 0 ,0, 1
#     while n < max:
#         yield b
#         a, b = b ,a+b
#         n += 1

# right
# for i in fab(10):
#     print(i)

# wrong
# for i in range(10):
#     print(fab(10))

# a = fab(3)
# for i in range(5):
#     print(a.__next__())
#
# for i in range(5):
#     print(fab(5).__next__())
