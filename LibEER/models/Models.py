from importlib import import_module


class LazyModelRegistry(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, tuple):
            module_name, attr_name = value
            value = getattr(import_module(module_name), attr_name)
            self[key] = value
        return value


Model = LazyModelRegistry({
    'DGCNN': ('models.DGCNN', 'DGCNN'),
    'CoralDgcnn': ('models.CoralDgcnn', 'CoralDgcnn'),
    'DannDgcnn': ('models.DannDgcnn', 'DannDgcnn'),
    'R2GSTNN': ('models.R2GSTNN', 'R2GSTNN'),
    'BiDANN': ('models.BiDANN', 'BiDANN'),
    'RGNN_official': ('models.RGNN_official', 'SymSimGCNNet'),
    'GCBNet': ('models.GCBNet', 'GCBNet'),
    'GCBNet_BLS': ('models.GCBNet_BLS', 'GCBNet_BLS'),
    'CDCN': ('models.CDCN', 'CDCN'),
    'DBN': ('models.DBN', 'DBN'),
    'STRNN': ('models.STRNN', 'STRNN'),
    'EEGNet': ('models.EEGNet', 'EEGNet'),
    'HSLT': ('models.HSLT', 'HSLT'),
    'ACRNN': ('models.ACRNN', 'ACRNN'),
    'TSception': ('models.TSception', 'TSception'),
    'MsMda': ('models.MsMda', 'MSMDA'),
    'FBSTCNet': ('models.FBSTCNet', 'PowerAndConneMixedNet'),
    'NSAL_DGAT': ('models.NSAL_DGAT', 'Domain_adaption_model'),
    'PRRL': ('models.PRRL', 'PRRL'),
    'svm': ('models.SVM', 'SVM'),
})
