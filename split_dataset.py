from utils.DataSplitter import *
from utils.DataReader import *

ds = Dataset()
sp = Splitter()

sp.build_testset(ds, "a")
#sp.build_testset(ds, "b")