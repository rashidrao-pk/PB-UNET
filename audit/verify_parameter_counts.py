#!/usr/bin/env python3
"""Static parameter-count check independent of TensorFlow."""

def conv_block(in_ch, out_ch):
    # two Conv2D layers including bias + two BatchNorm layers (4 params/channel each)
    return (9*in_ch*out_ch + out_ch + 4*out_ch
            + 9*out_ch*out_ch + out_ch + 4*out_ch), out_ch


def baseline(filters=(32,64,128), in_ch=3):
    total=0; ch=in_ch; skips=[]
    for f in filters:
        p,ch=conv_block(ch,f); total+=p; skips.append(ch)
    p,ch=conv_block(ch,filters[-1]); total+=p
    for f,s in zip(filters[::-1],skips[::-1]):
        ch += s; p,ch=conv_block(ch,f); total+=p
    return total + ch + 1


def proposed(filters=(32,64,128), in_ch=3):
    total=0; ch=in_ch; skips=[]
    for f in filters:
        p,ch=conv_block(ch,f); total+=p; skips.append(ch)
    rev=filters[::-1]; sr=skips[::-1]
    p,ch=conv_block(ch,filters[-1]); total+=p
    s1=[]
    for i in range(len(rev)-1):
        ch += sr[i]; p,ch=conv_block(ch,rev[i]); total+=p; s1.append(ch)
    s1=s1[::-1]; s2=[]
    for i in range(len(rev)-1):
        p,ch=conv_block(ch,rev[i]); total+=p; ch += s1[i]; s2.append(ch)
    s2=s2[::-1]
    for i in range(len(rev)-1):
        ch += s2[i]; p,ch=conv_block(ch,rev[i]); total+=p
    ch += sr[-1]
    return total + ch + 1

print('Baseline U-Net total params :', baseline())
print('Portable-Bridge total params:', proposed())
