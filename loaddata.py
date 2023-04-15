class defVar():
    def __init__(self,dataCols):
        self.dataCols = dataCols 

    def DummyVar(self):
        # var_list = self.dataCols[0:27]
        var_list = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
               'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',
               'x22', 'x23', 'x24', 'x25', 'x26', 'x27']
        return var_list

    def WghtVar(self):
        # wght_list = self.dataCols[27:54]
        # wght_list = ['x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37',
        #        'x38', 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47',
        #        'x48', 'x49', 'x50', 'x51', 'x52', 'x53', 'x54']
        wght_list = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
               'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',
               'x22', 'x23', 'x24', 'x25', 'x26', 'x27']
        return wght_list

    def YVar(self):
        # y_list = self.dataCols[54:55]
        y_list = ['y']
        return y_list


