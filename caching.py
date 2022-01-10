"""
a cache that stores the outputs of image-wide labeling functions
it only keeps one image at a time, thus it is emptied when changing image
"""
from functools import wraps
from typing import Any, Callable

global cache
cache = dict() # image

def store(image:Any, function_name:str):
    global cache
    cache[function_name] = image
def retrieve(function_name: str)->Any:
    global cache
    return cache.get(function_name,None)

def clear():
    global cache
    cache.clear()

def caching(func:Callable):
    def wrapper(*args,**kwargs):
        data = retrieve(func.__name__)
        if data is None:
            data = func(*args,**kwargs)
            store(data, func.__name__)
        return data
    return wrapper

def image_caching(func:Callable):
    """
    wraps a whole image preprocessor
    signature: f(image_data)
    whereas the actual arguments that you passded to snorkel should be [image_data,x,y],
    but this decorator modifies it
    """
    def wrapper(arg):
        print('image_caching wrapper')
        image,*args = arg
        output = retrieve(func.__name__)
        if output is  None:
            output = func(image)
            store(output,func.__name__)
        return [output, *args]
    return wrapper

def image_labeling_function(func):
    def wrapper(image, x, y):
        return
    return

def _preprocessor(pre_func):
    def _decorator_preprocessor(func):
        #   @wraps(func)
        #def wrapper(*args, **kwargs):
        def wrapper(img,x,y):
            x1 = pre_func(img)
            #x1 = pre_func(*args, **kwargs)
            return func(x1,x,y)
        return wrapper
    return _decorator_preprocessor

@caching
def preprocessor0(image):
    print('preprocessor0 called')
    return image + 'oui'


@_preprocessor(preprocessor0)
def l_f(image,x,y):
    print("l_f called")
    return image,x
if __name__ == '__main__':
    print(l_f('zae',0,0))
    print(l_f('zae',0,1))
    print(l_f('zae',1,1))
    clear()
    print(l_f('lol',1,1))
    print(l_f('lol',0,1))
    print(l_f('lol',0,0))
