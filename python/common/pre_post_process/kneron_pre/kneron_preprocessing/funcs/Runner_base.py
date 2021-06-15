from abc import ABCMeta, abstractmethod

class Param_base(object):
    @abstractmethod
    def update(self,**dic):
        raise NotImplementedError("Must override")

    def load_dic(self, key, **dic):
        if key in dic:
            param = eval('self.'+key)
            param = dic[key]

    def __str__(self):
        str_out = []
        return(' '.join(str_out))
  

class Common(Param_base):
    print_info = False
    model_size = [0,0]
    numerical_type = 'floating'

    def update(self, **dic):
        self.print_info = dic['print_info']
        self.model_size = dic['model_size']
        self.numerical_type = dic['numerical_type']
    
    def __str__(self):
        str_out = ['numerical_type:',str(self.numerical_type)]
        return(' '.join(str_out))
    
class Runner_base(metaclass=ABCMeta):
    common = Common()
    general = Param_base()
    floating = Param_base()
    hw = Param_base()

    def update(self, **kwargs):
        ## update param
        self.common.update(**kwargs['common'])
        self.general.update(**kwargs['general'])
        assert(self.common.numerical_type.lower() in ['floating', '520', '720'])
        if (self.common.numerical_type == 'floating'):
            if (self.floating.__class__.__name__ != 'Param_base'):
                self.floating.update(**kwargs['floating'])
        else:
            if (self.hw.__class__.__name__ != 'Param_base'):
                self.hw.update(**kwargs['hw'])

    def print_info(self):
        if (self.common.numerical_type == 'floating'):
            print(self, self.common, self.general, self.floating)
        else:
            print(self, self.common, self.general, self.hw)
        


        

