viewsh = 13;
viewsv = 13;
input2 = '/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/Bikes/';
%fileResults = fopen('Results/metrics_results.csv','w');
total_views = viewsh*viewsv;
%addpath('eval/')

results = []; % Initialize an empty array to store results
rates = [0.75 0.1 0.02 0.005];
for rate = rates    
    input1 = sprintf('../%.3g/', rate); % Use %.3g to format without unnecessary trailing zeros
    total_psnr_yuv = 0;
    total_psnr_y = 0;
    total_ssim = 0;
    file = dir(sprintf('../bikes_%.3g.comp',rate));
    filesize = file.bytes * 8; % File size in bits
    true_rate = filesize/(viewsh*viewsv * 625 * 434);
    for s = 1:(viewsh)
        for t = 1:(viewsv)
            filename = sprintf('%03d_%03d.ppm' , t, s);
            img1=imread([input1 filename]);
            img2=imread([input2 filename]);
            [psnr_y, ~, ~, psnr_yuv, ssim_y] = QM(img1, img2,10,10);
            total_psnr_yuv = total_psnr_yuv + psnr_yuv;
            total_psnr_y = total_psnr_y + psnr_y;
            total_ssim = total_ssim + ssim_y;
        end
    end
    mean_psnr_yuv = total_psnr_yuv / total_views;
    mean_psnr_y = total_psnr_y / total_views;
    mean_ssim = total_ssim / total_views;
    
    % Append the results for the current rate
    results = [results; true_rate, mean_psnr_y, mean_psnr_yuv, mean_ssim, rate]
end
% Write results to a CSV file with a header
header = {'Rate', 'Mean_PSNR_Y', 'Mean_PSNR_YUV', 'Mean_SSIM','Target_Rate'};
csvwrite_with_headers('metrics_results.csv', results, header);


% Helper function to write CSV with headers
function csvwrite_with_headers(filename, data, header)
    fid = fopen(filename, 'w');
    fprintf(fid, '%s\n', strjoin(header, ','));
    fclose(fid);
    dlmwrite(filename, data, '-append');
end