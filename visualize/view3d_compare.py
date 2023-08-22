import tkinter.filedialog

import numpy as np
from mayavi import mlab


def model2xyz(model):
    return model[:, 0], model[:, 1], model[:, 2]


def model2xyzs(model):
    return model[:, 0], model[:, 1], model[:, 2], model[:, -1]

def compare():
    while True:
        fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('所有文件', '.*'), ('文本文件', '.txt')])
        print(fns)
        if len(fns) == 0: exit(0)
        figure = mlab.figure(bgcolor=(1, 1, 1))
        flag=True
        for fn in fns:
            model1 = np.loadtxt(fn, delimiter=' ')
            model1 = model1[:1024]
            # without labels
            if model1.shape[-1]==2:
                model1=np.concatenate([model1,np.zeros([model1.shape[0],1])],axis=-1)
            x, y, z = model2xyz(model1)
            if flag:
                color=(0.3, 0.3, 1)
                flag=False
                mlab.points3d(x, y, z, color=color, figure=figure, scale_factor=0.05)
                # mlab.points3d(x, y, z, color=color, figure=figure, scale_factor=0.1)
            else:
                color=(1,0.3,0.3)
                mlab.points3d(x, y, z, color=color, figure=figure, scale_factor=0.05)
                # mlab.points3d(x, y, z, color=color, figure=figure, scale_factor=0.1)

        # with labels
        # x,y,z,s=model2xyzs(model1)
        # figure = mlab.figure(bgcolor=(1,1,1))
        # mlab.points3d(x,y,z,s,figure=figure,scale_mode="none",scale_factor=0.06)
        # # mlab.savefig(str(fn).split('.')[0]+".jpg",figure=figure)
        # mlab.show()
        mlab.show()

def compare_two():
    while True:
        fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('所有文件', '.*'), ('文本文件', '.txt')])
        print(fns)
        if len(fns) == 0: exit(0)
        figure1 = mlab.figure(bgcolor=(1, 1, 1))
        figure2 = mlab.figure(bgcolor=(1, 1, 1))
        if len(fns) is not 3:continue

        model1 = np.loadtxt(fns[0], delimiter=' ')
        model2 = np.loadtxt(fns[1], delimiter=' ')
        model3 = np.loadtxt(fns[2], delimiter=' ')

        x, y, z = model2xyz(model1)
        color1=(0.3, 0.3, 1)
        color2=(1,0.3,0.3)
        mlab.points3d(x, y, z, color=color1, figure=figure1, scale_factor=0.06)
        mlab.points3d(x, y, z, color=color1, figure=figure2, scale_factor=0.06)

        x,y,z=model2xyz(model2)
        mlab.points3d(x, y, z, color=color2, figure=figure1, scale_factor=0.06)

        x,y,z=model2xyz(model3)
        mlab.points3d(x, y, z, color=color2, figure=figure2, scale_factor=0.06)

        mlab.show()

def view_disp():
    while True:
        fns = tkinter.filedialog.askopenfilenames(title='选择文件', filetypes=[('所有文件', '.*'), ('文本文件', '.txt')])
        print(fns)
        if len(fns) != 2: exit(-1)
        figure = mlab.figure(bgcolor=(1, 1, 1))
        flag=True
        model1 = np.loadtxt(fns[0], delimiter=' ')
        model2 = np.loadtxt(fns[1], delimiter=' ')
        disp=model2-model1
        x,y,z=model2xyz(model1)
        u,v,w=model2xyz(disp)
        mlab.quiver3d(x,y,z,u,v,w,figure=figure,scale_factor=1)
        mlab.show()


if __name__ == '__main__':
    compare_two()
    # view_disp()
    # compare()
