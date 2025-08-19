
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

def viz_complete_blink_prop(data,row,srate):


    """

    TODO Viz

    https://stackoverflow.com/a/51928241/6446053

    :return:
    """


    xLabelString='T'
    fig, ax = plt.subplots(figsize=(8, 6))

    npad = 20
    preLimit = row['start_blink'] - npad
    postLimit = row['end_blink'] + npad

    idx_t = np.arange(preLimit, postLimit + 1)

    bTrace = data[idx_t]


    plt.plot(idx_t, bTrace,linestyle='-',marker='o',color='b',
             label='line with marker',alpha=0.7)
    plt.plot([idx_t[0], idx_t[-1]], [0, 0], "--", color="gray", lw=2,label='Y0')


    plt.plot([row['xLineCross_l'] , row['x_intersect']], [row['yLineCross_l'],  row['y_intersect']], "--", color="gray", lw=2)
    plt.plot([row['x_intersect'],row['xLineCross_r']], [row['y_intersect'],row['yLineCross_r']], "--", color="gray", lw=2)

    ## PLot key point
    plt.scatter([row['blinkBottomPoint_l_X'],row['blinkTopPoint_l_X']],
                [row['blinkBottomPoint_l_Y'],row['blinkTopPoint_l_Y']],
                marker='*', s=200,label='left_top_down_blink')

    plt.scatter([row['blinkBottomPoint_r_X'],row['blinkTopPoint_r_X']],
                [row['blinkBottomPoint_r_Y'],row['blinkTopPoint_r_Y']],
                marker='*', s=200,label='right_top_down_blink')



    plt.scatter(row['x_intersect'], row['y_intersect'],label='tent_point')


    plt.scatter([row['left_zero'], row['right_zero']], [0, 0], marker='d', s=100,label='zero crossing')
    plt.scatter(row['max_blink'], data[row['max_blink']],label='max Frame')

    plt.legend()
    ylabel='Signal(uv)'
    plt.xlabel(xLabelString)
    plt.ylabel(ylabel)
    bquality= 'Good'
    max_blink=row['max_blink']
    d=dict(fig=fig,
           blink_quality=bquality,
           maxFrames=max_blink)

    return d