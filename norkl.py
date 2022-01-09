from typing import Callable, Any, List, Dict

_LabelingFunction = Callable[[Any, int, int], bool]
_ImagePreprocessor = Callable[[Any], Any]

class LabelingFunction:
    def __init__(self, f:_LabelingFunction, preprocessors:List[_ImagePreprocessor]):
        self.f = f
        self.preprocessors = preprocessors
        self.cache = {}
        self.name = f.__name__

    def compute_image(self, image):
        _image = image
        for function in self.preprocessors:  # transform all the images
            _image = function(_image)
        return _image

    def cached_image(self,image):
        image_hash = hash(image)
        cached = self.cache.get(image_hash)
        if cached is not None:
            return cached
        self.cache[image_hash] = self.compute_image(image)
        return self.cache[image_hash]

    def __call__(self, image, x, y):
        processed_image = self.cached_image(image)
        return self.f(processed_image,x,y)

def labeling_function_decorator(preprocessors=[]):
    def wrapper(labeling_function):
        return LabelingFunction(labeling_function, preprocessors)
    return wrapper

class Applier:
    def __init__(self, labeling_functions: List[_LabelingFunction]):
        self.labeling_functions = labeling_functions

    def apply(self, image: Any, x: int, y: int) -> List[bool]:
        out = []
        for lf in self.labeling_functions:
            out.append(lf(image, x, y))
        return out

    def __call__(self, preprocessors: List[_ImagePreprocessor]=[]):
        def wrapper(f):
            wf = LabelingFunction(f, preprocessors)
            self.labeling_functions.append(wf)
            return wf
        return wrapper

def to_gray(image):
    print("preprocesssed to_gray")
    return image + "gray"

def to_color(image):
    print("preprocessed to_color")
    return image + "color"

def invert(image):
    print("preprocessed invert")
    return ''.join( reversed(image) )

@labeling_function_decorator(preprocessors=[to_gray])
def l_f0(image, x, y) -> bool:
    return f"{x},{y}:lf0:{image}"

@labeling_function_decorator(preprocessors=[to_color,invert])
def l_f1(image, x, y) -> bool:
    return f"{x},{y}:lf1:{image}"



if __name__ == '__main__':
    applier = Applier(labeling_functions=[l_f0,l_f1])
    img0 = "image"
    print(applier.apply(img0, 0, 0))
    print(applier.apply(img0, 1, 0))

    @applier(preprocessors=[invert])
    def lf_3(image,x,y):
        return f"{x},{y}:lf3:{image}"

    img0 = "picure"

    print(applier.apply(img0, 1, 1))
    print(applier.apply(img0, 0, 0))
    print(applier.apply(img0, 1, 0))
    print(applier.apply(img0, 1, 1))
    exit(0)
