class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        # backword用に保持しておく
        self.x = x
        self.y = y
        return x * y

    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backword(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

#
# MulLayer
#

# given
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
print("MulLayer ----- forward")
apple_price = mul_apple_layer.forward(apple, apple_num)
print("apple_price:", apple_price)
price = mul_tax_layer.forward(apple_price, tax)
print("price:", price)
print()

# backword
print("MulLayer ----- backword")
dprice = 1
dapple_price, dtax = mul_tax_layer.backword(dprice)
print("dapple_price:", dapple_price)
print("dtax:", dtax)
dapple, dapple_num = mul_apple_layer.backword(dapple_price)
print("dapple:", dapple)
print("dapple_num:", dapple_num)
print()


#
# + AddLayer
#

# given
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
print("MulLayer + AddLayer ----- forward")
apple_price = mul_apple_layer.forward(apple, apple_num)
print("apple_price:", apple_price)
orange_price = mul_orange_layer.forward(orange, orange_num)
print("orange_price:", orange_price)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
print("all_price:", all_price)
price = mul_tax_layer.forward(all_price, tax)
print("price w/ tax:", price)
print()

# backword
print("MulLayer + AddLayer ----- backword")
dprice = 1
dall_price, dtax = mul_tax_layer.backword(dprice)
print("dall_price:", dall_price)
print("dtax:", dtax)
dapple_price, dorange_price = add_apple_orange_layer.backword(dall_price)
print("dapple_price:", dapple_price)
print("dorange_price:", dorange_price)
dapple, dapple_num = mul_apple_layer.backword(dapple_price)
print("dapple:", dapple)
print("dapple_num:", dapple_num)
dorange, dorange_num = mul_orange_layer.backword(dorange_price)
print("dorange:", dorange)
print("dorange_num:", dorange_num)

