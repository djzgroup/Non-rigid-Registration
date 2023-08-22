import utils.LoaderFish as Fish
import numpy as np
dataset = Fish.PointRegDataset(deform_level=1.5, total_data=10)
zero=np.zeros([91,1])
for idx in range(len(dataset)):
    data=dataset[idx]
    target,source,_,_=data
    target,source=target.T,source.T
    np.savetxt(idx.__str__()+"tgt.txt",np.concatenate([target,zero],axis=-1))
    # np.savetxt(idx.__str__()+"src.txt",np.concatenate([source,zero],axis=-1))
