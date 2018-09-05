import matplotlib.pyplot as plt
import numpy as np

def truth_vs_predict_mask(df, output_file):
    max_images = 50
    grid_width = 5
    grid_height = int(max_images / grid_width) * 3
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width * 4, grid_height * 4))
    for i in range(max_images):
        ax_image = axs[int(i / grid_width) * 3, i % grid_width]
        ax_image.imshow(df['img_data'][i], cmap="Greys")
        ax_image.set_title("Image {0}\nDepth: {1}".format(df['img_name'][i], df['z'][i]))
        ax_image.set_yticklabels([])
        ax_image.set_xticklabels([])
        ax_mask = axs[int(i / grid_width) * 3 + 1, i % grid_width]
        ax_mask.imshow(df['img_data'][i], cmap="Greys")
        ax_mask.imshow(df['mask_data'][i], alpha=0.9, cmap="Greens")
        ax_mask.set_title("Mask {0}\nCoverage: {1}, IOU {2}".format(df['img_name'][i],
                                                                    round(df['mask_coverage_ratio'][i], 2),
                                                                    round(df['iou'][i], 2)))
        ax_mask.set_yticklabels([])
        ax_mask.set_xticklabels([])
        ax_pred = axs[int(i / grid_width) * 3 + 2, i % grid_width]
        ax_pred.imshow(df['img_data'][i], cmap="Greys")
        ax_pred.imshow(df['pred_mask_data'][i], alpha=0.9, cmap="Blues")
        ax_pred.set_title("Predict {0}\nCoverage: {1}, PREC {2}".format(df['img_name'][i],
                                                                        round(df['pred_mask_coverage_ratio'][i], 2),
                                                                        round(df['prec'][i], 2)))
        ax_pred.set_yticklabels([])
        ax_pred.set_xticklabels([])
    plt.savefig(output_file)
