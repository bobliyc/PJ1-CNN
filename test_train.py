import os
import numpy as np
import gzip
import pickle
from struct import unpack
import matplotlib.pyplot as plt

# ——— Config ———
use_cnn     = False           # False: MLP, True: CNN
nHidden     = [784, 256, 128, 10]
lr          = 0.1
momentum    = 0.9
reg         = 1e-4
dropout_p   = 0.0             # no dropout for speed
epochs      = 5
batch_size  = 128
milestones  = [3, 4]          # LR drops at epoch 3 and 4

# ——— Layers ———
class Linear:
    def __init__(self, in_f, out_f, reg=0.0):
        self.W = np.random.randn(in_f,out_f)*np.sqrt(2/(in_f+out_f))
        self.b = np.zeros(out_f)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.reg = reg
    def forward(self,x):
        self.x_shape = x.shape
        self.x_flat = x.reshape(x.shape[0],-1)
        return self.x_flat @ self.W + self.b
    def backward(self,g):
        # gradient + L2
        self.grad_W[:] = self.x_flat.T @ g + self.reg*self.W
        self.grad_b[:] = g.sum(axis=0)
        dx = g @ self.W.T
        return dx.reshape(self.x_shape)

class ReLU:
    def forward(self,x):
        self.mask = (x>0)
        return x*self.mask
    def backward(self,g):
        return g*self.mask

class SoftmaxCE:
    @staticmethod
    def forward(logits,y):
        z = logits - logits.max(axis=1,keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1,keepdims=True)
        N = logits.shape[0]
        loss = -np.log(p[np.arange(N),y]+1e-12).mean()
        SoftmaxCE.cache = (p,y,N)
        return loss,p
    @staticmethod
    def backward():
        p,y,N = SoftmaxCE.cache
        g = p.copy()
        g[np.arange(N),y] -= 1
        return g/N

class SGD_Momentum:
    def __init__(self,params,lr=0.1,momentum=0.9):
        self.params,self.lr,self.m = params,lr,momentum
        self.v = [np.zeros_like(p['param']) for p in params]
    def step(self):
        for i,p in enumerate(self.params):
            self.v[i] = self.m*self.v[i] - self.lr*p['grad']
            p['param'] += self.v[i]

class MultiStepLR:
    def __init__(self,opt,milestones,gamma=0.1):
        self.opt,self.ms,self.gamma = opt,set(milestones),gamma
        self.epoch=0
    def step(self):
        self.epoch += 1
        if self.epoch in self.ms:
            self.opt.lr *= self.gamma

# ——— CNN layers ———
class Conv2D:
    def __init__(self,in_c,out_c,k,stride=1,pad=1,reg=0.0):
        self.in_c,self.out_c,self.k = in_c,out_c,k
        self.s,self.p = stride,pad
        limit = np.sqrt(6/(in_c*k*k + out_c*k*k))
        self.W = np.random.uniform(-limit,limit,(out_c,in_c,k,k))
        self.b = np.zeros(out_c)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.reg = reg
    def forward(self,x):
        N,C,H,W = x.shape
        outH = (H+2*self.p-self.k)//self.s + 1
        outW = (W+2*self.p-self.k)//self.s + 1
        xp = np.pad(x,((0,0),(0,0),(self.p,self.p),(self.p,self.p)),'constant')
        out = np.zeros((N,self.out_c,outH,outW))
        self.x,self.xp = x,xp
        for n in range(N):
            for oc in range(self.out_c):
                for i in range(outH):
                    for j in range(outW):
                        h0,w0 = i*self.s, j*self.s
                        win = xp[n,:,h0:h0+self.k,w0:w0+self.k]
                        out[n,oc,i,j] = np.sum(win*self.W[oc]) + self.b[oc]
        return out
    def backward(self,g):
        N,_,outH,outW = g.shape
        dxp = np.zeros_like(self.xp)
        self.grad_W.fill(0)
        self.grad_b.fill(0)
        for n in range(N):
            for oc in range(self.out_c):
                for i in range(outH):
                    for j in range(outW):
                        h0,w0 = i*self.s, j*self.s
                        win = self.xp[n,:,h0:h0+self.k,w0:w0+self.k]
                        grad = g[n,oc,i,j]
                        self.grad_W[oc] += win * grad
                        self.grad_b[oc] += grad
                        dxp[n,:,h0:h0+self.k,w0:w0+self.k] += self.W[oc]*grad
        # L2
        self.grad_W += self.reg*self.W
        if self.p>0:
            return dxp[:,:,self.p:-self.p,self.p:-self.p]
        return dxp

class MaxPool2D:
    def __init__(self,k=2,stride=2):
        self.k,self.s = k,stride
    def forward(self,x):
        N,C,H,W = x.shape
        outH = (H-self.k)//self.s + 1
        outW = (W-self.k)//self.s + 1
        out = np.zeros((N,C,outH,outW))
        self.x,self.mask = x,{}
        for n in range(N):
            for c in range(C):
                for i in range(outH):
                    for j in range(outW):
                        h0,w0 = i*self.s, j*self.s
                        win = x[n,c,h0:h0+self.k,w0:w0+self.k]
                        m = win.max()
                        out[n,c,i,j] = m
                        self.mask[(n,c,i,j)] = (win==m)
        return out
    def backward(self,g):
        N,C,outH,outW = g.shape
        dx = np.zeros_like(self.x)
        for n in range(N):
            for c in range(C):
                for i in range(outH):
                    for j in range(outW):
                        h0,w0 = i*self.s, j*self.s
                        mask = self.mask[(n,c,i,j)]
                        dx[n,c,h0:h0+self.k,w0:w0+self.k] += g[n,c,i,j]*mask
        return dx

# ——— Models ———
class MLP:
    def __init__(self,sizes,reg=0.0):
        self.layers = []
        for i in range(len(sizes)-1):
            self.layers.append(Linear(sizes[i],sizes[i+1],reg))
            if i < len(sizes)-2:
                self.layers.append(ReLU())
    def forward(self,x,train=True):
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out
    def backward(self,g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g
    def parameters(self):
        ps = []
        for l in self.layers:
            if isinstance(l,Linear):
                ps += [{'param':l.W,'grad':l.grad_W},
                       {'param':l.b,'grad':l.grad_b}]
        return ps

class SimpleCNN:
    def __init__(self,reg=0.0):
        self.conv1 = Conv2D(1,16,3,1,1,reg)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(16,32,3,1,1,reg)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        self.fc    = Linear(32*7*7,10,reg)
    def forward(self,x,train=True):
        N=x.shape[0]; x=x.reshape(N,1,28,28)
        x=self.pool1.forward(self.relu1.forward(self.conv1.forward(x)))
        x=self.pool2.forward(self.relu2.forward(self.conv2.forward(x)))
        x=x.reshape(N,-1)
        return self.fc.forward(x)
    def backward(self,g):
        g=self.fc.backward(g); g=g.reshape(-1,32,7,7)
        g=self.conv2.backward(self.relu2.backward(self.pool2.backward(g)))
        g=self.conv1.backward(self.relu1.backward(self.pool1.backward(g)))
        return g
    def parameters(self):
        ps=[]
        for layer in [self.conv1,self.conv2,self.fc]:
            ps += [{'param':layer.W,'grad':layer.grad_W},
                   {'param':layer.b,'grad':layer.grad_b}]
        return ps

# ——— Data ———
def load_mnist(ip,lp):
    with gzip.open(ip,'rb') as f:
        _,n,r,c = unpack('>4I',f.read(16))
        imgs = np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r*c)
    with gzip.open(lp,'rb') as f:
        _,n = unpack('>2I',f.read(8))
        lbls = np.frombuffer(f.read(),dtype=np.uint8)
    return imgs,lbls

# ——— Main Training ———
if __name__=='__main__':
    np.random.seed(309)
    imgs,lbls = load_mnist(
        './dataset/MNIST/train-images-idx3-ubyte.gz',
        './dataset/MNIST/train-labels-idx1-ubyte.gz'
    )
    idx = np.random.permutation(len(lbls))
    imgs,lbls = imgs[idx]/255.0, lbls[idx]

    # full split
    X_train,y_train = imgs[10000:], lbls[10000:]
    X_val,  y_val   = imgs[:10000], lbls[:10000]

    # build
    model = SimpleCNN(reg) if use_cnn else MLP(nHidden, reg)
    opt   = SGD_Momentum(model.parameters(), lr, momentum)
    sched = MultiStepLR(opt, milestones, gamma=0.1)

    best_acc = 0
    for ep in range(1, epochs+1):
        perm = np.random.permutation(len(y_train))
        X_train, y_train = X_train[perm], y_train[perm]
        for i in range(0, len(y_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            logits,_ = SoftmaxCE.forward(model.forward(xb, True), yb)
            loss,_   = SoftmaxCE.forward(model.forward(xb, True), yb)
            grad      = SoftmaxCE.backward()
            model.backward(grad)
            opt.step()
        sched.step()

        # evaluate
        p_tr = SoftmaxCE.forward(model.forward(X_train[:5000],False), y_train[:5000])[1]
        tr_acc = (p_tr.argmax(1)==y_train[:5000]).mean()
        p_v  = SoftmaxCE.forward(model.forward(X_val,False), y_val)[1]
        v_acc = (p_v.argmax(1)==y_val).mean()

        print(f"Epoch {ep}/{epochs} — TrainAcc: {tr_acc*100:.2f}%  ValAcc: {v_acc*100:.2f}%")

        # save best
        if v_acc > best_acc:
            best_acc = v_acc
            os.makedirs('best_models', exist_ok=True)
            with open('best_models/best_model.pkl','wb') as f:
                pickle.dump(model, f)

    # final plot
    plt.plot(range(1,epochs+1), [best_acc*100]*epochs, label='Best ValAcc')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy (%)')
    plt.title('Final Validation Accuracy'); plt.legend(); plt.show()
