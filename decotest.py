def deco1(f):
    def wrapper(*args):
        print("deco1")
        return f(*args)
    return wrapper

def deco2(f):
    def wrapper(*args):
        print("deco2")
        return f(*args)
    return wrapper

@deco1
@deco2
@deco2
def f1(x):
    return x+1

if __name__ == '__main__':
    print(f1(1))



