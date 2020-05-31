import my_file_handler as mfh

class Parameter(mfh.SParameter):
    def __init__(self, a=0., b=0.):
        super().__init__()
        
        self.pdict['a'] = a
        self.pdict['b'] = b


param = Parameter()
print(param.get_filename())

sfh = mfh.SimuFileHandler('../TemporalEvolGame/Interact')
print(sfh.get_filepath(param))
print(type(sfh.get_filepath(param)))