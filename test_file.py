if args.num_views == 2:
    ckpt_dir = f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/{scene_tmp}_two_views"
    view_scale = np.load(f"/scratch/ssd002/projects/anagh_neurips/data/{scene_tmp}_data/{scene_tmp}_jsons/two_views/max.npy")
elif args.num_views == 3:
    ckpt_dir = f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/{scene_tmp}_three_views"
    view_scale = np.load(f"/scratch/ssd002/projects/anagh_neurips/data/{scene_tmp}_data/{scene_tmp}_jsons/three_views/max.npy")
elif args.num_views == 5:
    ckpt_dir = f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/{scene_tmp}_five_views"
    view_scale = np.load(f"/scratch/ssd002/projects/anagh_neurips/data/{scene_tmp}_data/{scene_tmp}_jsons/five_views/max.npy")
elif args.num_views == 10:
    ckpt_dir = f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/{scene_tmp}_ten_views"
    view_scale = np.load(f"/scratch/ssd002/projects/anagh_neurips/data/{scene_tmp}_data/{scene_tmp}_jsons/ten_views/max.npy")
else:
    print("unknown number of views!!")
    # render transients and other outputs from network
    # for i in range(len(test_dataset)):
for k in range(3):
    if args.scene == "hotdog" and args.split == "test3":
        i = k + 1
    else:
        i = k

    exr_depth = get_gt_depth(positions["frames"][i], test_dataset.camtoworlds[i].cpu().numpy(), data_root_fp)
    ind = int(positions["frames"][i]["file_path"].split("_")[-1])
    print(f"test image {ind}")

    rgb = np.zeros((img_shape[0], img_shape[1], n_bins, 3))
    depth = np.zeros((img_shape[0], img_shape[1]))
    weights_sum = 0
    for j in range(rep_number):
        print(f"rep {j}")
        # if j == 0:
        #     test_dataset.testing = False
        # else:
        #     test_dataset.testing = True

        data = test_dataset[i]
        pixels = data["pixels"].detach().cpu().numpy()
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        # if j != 0:
        #     sample_weights = data["weights"]
        sample_weights = data["weights"]
        del data
        # rays = namedtuple_map(lambda r: r[::4, ::4], rays)

        out = render_image_two_bounce(
            radiance_field,
            occupancy_grid,
            rays,
            radiance_field.aabb,
            # rendering=rendering,
            # rgb_sigma_fn=rgb_sigma_fn,
            # rendering options
            near_plane=None,
            far_plane=None,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=0,
            alpha_thre=0,
            n_bins=n_bins,
            exposure_time=exposure_time,
            step=step,
            return_sample_histogram=False,
            chunk=4096
        )


        depth += (out["depths"]*sample_weights[:, None]*(out["opacities"])).reshape(img_shape[0], img_shape[1]).detach().cpu().numpy()
        rgb += (out["colors"] * sample_weights[:, None]).reshape(img_shape[0], img_shape[1], n_bins,
                                                                    3).detach().cpu().numpy()
        weights_sum += sample_weights.detach().cpu().numpy()


        del out

    rgb = rgb / weights_sum.reshape(img_shape[0], img_shape[1], 1, 1)
    depth = depth / weights_sum.reshape(img_shape[0], img_shape[1])

    pixels = pixels.reshape(img_shape[0], img_shape[1], n_bins, 3)

    # (a) rendering depth against gt
    mask = (pixels.sum((-1, -2)) > 0)
    error_gt_rnd = np.abs(exr_depth - depth)
    percentage_error = (error_gt_rnd / (exr_depth + 1e-10))[mask]
    l1_errors_gt_rnd.append((error_gt_rnd * mask).mean())
    l1_errors_gt_rnd_pct.append(100 * percentage_error.mean())

    print(
        f"Error between ground truth depth and our rendered depth {error_gt_rnd[mask].mean()} ({100 * percentage_error.mean()}%)")

    error_gt_rnd = np.abs(exr_depth - depth)**2
    percentage_error = (error_gt_rnd / (exr_depth + 1e-10))[mask]
    MSE_errors_gt_rnd.append((error_gt_rnd * mask).mean())
    MSE_errors_gt_rnd_pct.append(100 * percentage_error.mean())


    # (b) depth from log matched filter on our transient against gt
    lg_depth = get_depth_from_transient(rgb, exposure_time=exposure_time, tfilter_sigma=3)
    error_gt_lm_ours = np.abs(exr_depth - lg_depth)
    percentage_error = (error_gt_lm_ours / (exr_depth + 1e-10))[mask]
    l1_errors_gt_lm_ours.append((error_gt_lm_ours * mask).mean())
    l1_errors_gt_lm_ours_pct.append(100 * percentage_error.mean())
    print(
        f"Error between ground truth depth and log matched filter on our transient {(error_gt_lm_ours[mask]).mean()} ({100 * percentage_error.mean()}%)")

    plt.imsave(f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_{args.step}_test{ind}_lg_depth.png",
                    lg_depth, cmap='inferno', vmin=2.5, vmax=5.5)
    np.save(f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_{args.step}_test{ind}_lg_depth.png", lg_depth)

    error_gt_lm_ours = np.abs(exr_depth - lg_depth)**2
    percentage_error = (error_gt_lm_ours / (exr_depth + 1e-10))[mask]
    MSE_errors_gt_lm_ours.append((error_gt_lm_ours * mask).mean())
    MSE_errors_gt_lm_ours_pct.append(100 * percentage_error.mean())

    # (b) depth from log matched filter on our transient against gt
    gt_depth = get_depth_from_transient(pixels, exposure_time=exposure_time, tfilter_sigma=3)
    error_gt_lm_gt = np.abs(exr_depth - gt_depth)
    percentage_error = (error_gt_lm_gt / (exr_depth + 1e-10))[mask]
    l1_errors_gt_lm_gt.append((error_gt_lm_gt * mask).mean())
    l1_errors_gt_lm_gt_pct.append(100 * percentage_error.mean())
    print(
        f"Error between ground truth depth and log matched filter on gt transient {error_gt_lm_gt[mask].mean()} ({100 * percentage_error.mean()}%)")

    error_gt_lm_gt = np.abs(exr_depth - gt_depth)**2
    percentage_error = (error_gt_lm_gt / (exr_depth + 1e-10))[mask]
    MSE_errors_gt_lm_gt.append((error_gt_lm_gt * mask).mean())
    MSE_errors_gt_lm_gt_pct.append(100 * percentage_error.mean())

    #if ind == 7:
    #transient = torch.from_numpy(rgb).to_sparse()
    #torch.save(transient, f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_{args.step}_test{ind}_transient.pt")
    #continue
    # (c) psnr on rendered images (need to figure out how to gamma correct and all tho)
    #view_scale = 100
    rgb_image = rgb.sum(axis=-2)*view_scale/dataset_scale
    # rgb_image = (np.clip(rgb_image / 20, 0, 1) ** (1 / 2.2))
    # data_image = (np.clip(pixels.sum(-2) / 20, 0, 1) ** (1 / 2.2))
    rgb_image = np.clip(rgb_image, 0, 1) ** (1 / 2.2)
    data_image = (pixels.sum(-2)*test_dataset.max.numpy()/dataset_scale) ** (1 / 2.2)

    plt.imsave(f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_{args.step}_test{ind}_depth.png",
                    depth, cmap='inferno', vmin=2.5, vmax=5.5)
    np.save(f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_{args.step}_test{ind}_depth.png", depth)

    imageio.imwrite(f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_{args.step}_test{ind}_RGB.png",
                    (rgb_image*255.0).astype(np.uint8))

    imageio.imwrite(f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_{args.step}_test{ind}_RGB_gt.png",
                    (data_image*255.0).astype(np.uint8))

    mse_ = F.mse_loss(torch.from_numpy(data_image), torch.from_numpy(rgb_image))
    mses.append(mse_)
    print(f"Image mse {mse_}")

    psnr_ = psnr_fn(data_image, rgb_image)
    psnrs.append(psnr_)
    print(f"Image psnr {psnr_}")

    ssim_, _ = structural_similarity(data_image, rgb_image, full=True, channel_axis=2)
    ssims.append(ssim_)
    print(f"Image ssim {ssim_}")

    lpips_ = loss_fn_vgg(torch.from_numpy(data_image * 2 - 1).unsqueeze(-1).permute((3, 2, 0, 1)).to(torch.float32),
                            torch.from_numpy(rgb_image * 2 - 1).unsqueeze(-1).permute((3, 2, 0, 1)).to(torch.float32))
    lpips_ = lpips_.detach().cpu().numpy().flatten()[0]
    lpipss.append(lpips_)
    print(f"Image LPIPS {lpips_}")

    # (d) l2 or l1 on rendered transient
    rgb_norm = rgb / (rgb.sum(-2)[:, :, None, :] + 1e-10)
    pixels_norm = pixels / (pixels.sum(-2)[:, :, None, :] + 1e-10)
    kl_div = (pixels_norm * np.log(pixels_norm / (rgb_norm + 1e-19) + 1e-10)).sum((-1, -2))
    kl_div = kl_div[mask].mean()
    divs.append(kl_div)
    print(f"Transient kl divergence {kl_div}")

    print("-----")

    print(f"Average PSNR: {sum(psnrs) / len(psnrs)}")
    print(f"Average SSIM: {sum(ssims) / len(ssims)}")
    print(f"Average LPIPS: {sum(lpipss) / len(lpipss)}")

    print(f"Average errors_gt_rnd: {sum(l1_errors_gt_rnd) / len(l1_errors_gt_rnd)}")
    print(f"Average errors_gt_lm_ours: {sum(l1_errors_gt_lm_ours) / len(l1_errors_gt_lm_ours)}")
    print(f"Average errors_gt_lm_gt: {sum(l1_errors_gt_lm_gt) / len(l1_errors_gt_lm_gt)}")

    np.savetxt(f"/scratch/ssd002/projects/anagh_neurips/ablation_no_hdr/results/{args.scene}_{args.num_views}_end{ind}.txt",
               np.stack([np.array(mses), np.array(psnrs), np.array(ssims), np.array(lpipss), np.zeros((len(mses))),
                         np.array(l1_errors_gt_rnd), np.array(l1_errors_gt_lm_ours), np.array(l1_errors_gt_lm_gt),
                         np.array(l1_errors_gt_rnd_pct), np.array(l1_errors_gt_lm_ours_pct),
                         np.array(l1_errors_gt_lm_gt_pct), np.zeros((len(mses))),
                         np.array(MSE_errors_gt_rnd), np.array(MSE_errors_gt_lm_ours), np.array(MSE_errors_gt_lm_gt),
                         np.array(MSE_errors_gt_rnd_pct), np.array(MSE_errors_gt_lm_ours_pct),
                         np.array(MSE_errors_gt_lm_gt_pct), np.zeros((len(mses))),
                         np.array(divs)], axis=0))