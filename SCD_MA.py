import numpy as np
import matplotlib.pyplot as plt
from pywt import dwt2
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA


def SCD_MA(img=None):
   # clc(mstring('close'), mstring('all'))
#%%
   # Load image initialize variables
   Level = 5
   Io = 255
   NumBands = 20

   # addpath(mstring('FastICA_25'))

   h, w, _ = img.shape
   Statcount = 1
   StatMesur = np.zeros((3, Level * 4))

   # Reshape I into 3*(h*w)
   B = img[:, :, 2]
   G = img[:, :, 1]
   R = img[:, :, 0]
   RGB = np.array([R.T.reshape(( h * w)), G.T.reshape(( h * w)), B.T.reshape(( h * w))])  # LT: remove dim 1, en blijkbaar levert dit een getransposed resultaat op in matlab
   OD_M = - np.log((RGB + 1) / 255)

   # optical density of images
   OD = -np.log((img + 1) / 255)

   # set fillters and apply DWT
   wname = 'db8'

   # initialize Approximation bands with OD channels
   A1 = OD[:, :, 0]
   A2 = OD[:, :, 1]
   A3 = OD[:, :, 2]
   Bands = []
   # Bands = cell(Level, 4)
#%%
   for i in range(0, Level): 
      #% Generate Bands
      A1, (H1, V1, D1) = dwt2(A1, wname)
      A2, (H2, V2, D2) = dwt2(A2, wname)
      A3, (H3, V3, D3) = dwt2(A3, wname)

      BandH = A1.shape[0]
      BandW = A1.shape[1]
      LL = np.empty((BandH, BandW, 3))
      LH = np.empty((BandH, BandW, 3))
      HL = np.empty((BandH, BandW, 3))
      HH = np.empty((BandH, BandW, 3))

      # concatenate subbands based on colour channel
      LL[:, :, 0] = A1
      LL[:, :, 1] = A2
      LL[:, :, 2] = A3

      LH[:, :, 0] = H1
      LH[:, :, 1] = H2
      LH[:, :, 2] = H3

      HL[:, :, 0] = V1
      HL[:, :, 1] = V2
      HL[:, :, 2] = V3

      HH[:, :, 0] = D1
      HH[:, :, 1] = D2
      HH[:, :, 2] = D3

      LevelBands = [LL, LH, HL, HH]
      Bands.append(LevelBands)

      # Bands[i, 1) = LL
      # Bands(i, 2) = LH;
      # Bands(i, 3) = HL
      # Bands(i, 4) = HH;

      #% Show concatenated bands
      #ShowSubbands(LL,LH,HL,HH,i);

      #% Normalize bands to have zero mean and unit variance
      LL_1 = (LL - np.mean(LL)) / np.std(LL)
      LH_1 = (LH - np.mean(LH)) / np.std(LH)
      HL_1 = (HL - np.mean(HL)) / np.std(HL)
      HH_1 = (HH - np.mean(HH)) / np.std(HH)

      #% Compute Non-Gaussian Messures
      NumEle = LL_1.size
      Kor = np.array([kurtosis(LL_1.flatten(),fisher=False)-3, #LT: blijkbaar gebruikt matlab standaard Pearsons' kurtosis en Python Fishers'... 
                      kurtosis(LH_1.flatten(),fisher=False)-3,
                      kurtosis(HL_1.flatten(),fisher=False)-3,
                      kurtosis(HH_1.flatten(),fisher=False)-3])
      Kor = np.abs(Kor)
      z = 0
      for s in range(Statcount, Statcount+4): #in range(start,stop) stopt Matlab OP stop, en python VOOR stop
         StatMesur[0, s-1] = Kor[z] #LT: changed Kor[0] to Kor[z]
         StatMesur[1, s-1] = i+1
         StatMesur[2, s-1] = z+1
         z = z + 1

      Statcount = Statcount + 4
#%%
   # Sort Kurtosis matrix
   # from pdb import set_trace; set_trace()
   d2 = np.argsort(StatMesur[0])
   d2 = d2[::-1]
   StatMesur = StatMesur[:, d2]
#%%
   #% Concatenate Subbands
   Coff = Bands[0][0]  # Bands{level,band}
   r, c, _ = Coff.shape
   B = Coff[:, :, 2]
   G = Coff[:, :, 1]
   R = Coff[:, :, 0]
   Coff = np.array([np.reshape(R.T, ( r * c)),  #LT: remove dim of 1
                     np.reshape(G.T, ( r * c)), #En ook hier moet je transposen
                     np.reshape(B.T, ( r * c))])

   FinalSignal = Coff # need to be changed

   for i in range(0, NumBands): #LT: deze ging dus ook een te weinig
      Coff = Bands[int(StatMesur[1, i]-1)][int(StatMesur[2, i]-1)]            #Bands{level,band}
      r, c, _ = Coff.shape
      B = Coff[:, :, 2]
      G = Coff[:, :, 1]
      R = Coff[:, :, 0]
      Coff = np.array([np.reshape(R.T, (r * c)),   #LT: remove dim of 1
                        np.reshape(G.T, ( r * c)),
                        np.reshape(B.T, ( r * c))])

      FinalSignal = np.hstack((FinalSignal, Coff))

   
   cov = np.cov(FinalSignal, rowvar=True)   # cov is (N, N)
   # singular value decomposition
   U,S,V = np.linalg.svd(cov)     # U is (N, N), S is (N,)
   # build the ZCA matrix
   epsilon = 1e-16
   zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S +epsilon)), U.T))
   # transform the image data       zca_matrix is (N,N)
   zca = np.dot(zca_matrix, FinalSignal)    # zca is (N, 3072)

    #% apply ICA  

   fastica = FastICA(n_components=3,algorithm='deflation',max_iter=200)
   
   
   # LT: in matlab moet je matrix vorm [channels,samples] hebben, bij python andersom..
   # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
   fastica = fastica.fit(zca.T)  #LT: volgens de doc moet je m zo gebruiken.. duurt heel lang :(
   A = fastica.components_
   # fastica(n_components=3, FinalSignal, mstring('numOfIC'), 3)            #W{1}{1} is 3*3 matrix (each row for one source

   #% Compute OD and density image and stain matrix
   Ref_Vecs = np.abs(A)

   #% normalize stain vector
   for z in range(0, 3): #LT: eentje te weinig
      # normalise vector length
      length = np.sqrt((Ref_Vecs[0, z] * Ref_Vecs[0, z]) + (Ref_Vecs[1, z] * Ref_Vecs[1, z]) + (Ref_Vecs[2, z] * Ref_Vecs[2, z]))
      if length != 0.0:
         Ref_Vecs[0, z] = Ref_Vecs[0, z] / length
         Ref_Vecs[1, z] = Ref_Vecs[1, z] / length
         Ref_Vecs[2, z] = Ref_Vecs[2, z] / length

   #% sort to start with H
   Temp = np.copy(Ref_Vecs[:,:])
   c = np.argmax(Temp[0, :])
   Ref_Vecs[:, 0] = np.copy(Temp[:, c][:])  #LT: deepcopy is done implicitely in matlab
   Temp[:, c] = np.nan

   c = np.nanargmin(Temp[0, :])
   Ref_Vecs[:, 1] = np.copy(Temp[:, c][:])
   Temp[:, c] = np.nan
   c = np.nanargmin(Temp[0,:])
   Ref_Vecs[:, 2] = np.copy(Temp[:, c][:])
   
   #% compute density matrix and show results
   # d = (Ref_Vecs); print d
   # OD_M
   d =  np.linalg.lstsq(Ref_Vecs,OD_M,rcond=-1)[0]   #LT: matlabs A\B geeft de oplossing x waarvoor Ax=B 

   #d(d<0)=0;
   #% Show results
   H = Io * np.exp(np.outer(-Ref_Vecs[:, 0], d[0, :]))    #LT: changed * to matrix product! In matlab this is implicit for all operations
   H = H.reshape(3,h,w)
   H = np.uint8(H)
   
   H = np.transpose(H,(1,2,0))  #LT: ok this is a weird thing, but the order of dimensions is wrong for imshow

   E = Io * np.exp(np.outer(-Ref_Vecs[:, 1], d[1, :]))
   E = E.reshape(3, h, w)
   E = np.uint8(E)
   E = np.transpose(E,(1,2,0))

   Bg = Io * np.exp(np.outer(-Ref_Vecs[:, 2], d[2, :]))
   Bg = Bg.reshape(3, h, w)
   Bg = np.uint8(Bg)
   Bg = np.transpose(Bg,(1,2,0))
   
   plt.imshow(H)
   
   plt.imsave('H.PNG',H)
   plt.imsave('E.png',E)
   plt.imsave('Bg.png',Bg)
#   
#   
#   im = Image.fromarray(H)
#   im.save("H.png")
#   
#   im = Image.fromarray(E)
#   im.save("E.png")
#
#   im = Image.fromarray(Bg)
#   im.save("Bg.png")
#
   #% Show seperated stain for the sample image
   # figure
   # subplot(141)
   # imagesc(I)
   # axis("off")
   # title(mstring('Source'))

   # set(gcf, mstring('units'), mstring('normalized'), mstring('outerposition'), mcat([0, 0, 1, 1]))

   # subplot(142)
   # imagesc(H)
   # axis("off")
   # title(mstring('S1'))

   # set(gcf, mstring('units'), mstring('normalized'), mstring('outerposition'), mcat([0, 0, 1, 1]))

   # subplot(143)
   # imagesc(E)
   # axis("off")
   # title(mstring('S2'))

   # set(gcf, mstring('units'), mstring('normalized'), mstring('outerposition'), mcat([0, 0, 1, 1]))

   # subplot(144)
   # imagesc(Bg)
   # axis("off")
   # title(mstring('S3'))

   # set(gcf, mstring('units'), mstring('normalized'), mstring('outerposition'), mcat([0, 0, 1, 1]))

from PIL import Image
img = Image.open('Data/RGB_images/Breast/1/01/01.png')
img = np.array(img, dtype=np.float32)
SCD_MA(img) 
