
def CreateDataLoader(opt):
    data_loader = None
    if opt.align_data == 1:
        from data.aligned_data_loader import AlignedDataLoader
        data_loader = AlignedDataLoader()
    elif opt.align_data == 2:
        from data.unaligned_data_loader import UnalignedDataLoader
        data_loader = UnalignedDataLoader()
    else:
        from data.three_aligned_data_loader import ThreeAlignedDataLoader
        data_loader = ThreeAlignedDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
