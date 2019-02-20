%   Reliable Patch Trackers: Robust Visual Tracking by Exploiting Reliable Patches
%
%   Yang Li, 2015
%   http://ihpdep.github.io
%
%   This is the research code of our RPT tracker. You can check the
%   details in our CVPR paper.

function [positions,targetSize, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)

    randn('seed',0);rand('seed',0);
    
% the static parameters for all parts
    param={};
    param.padding=padding;
    param.kernel=kernel;
    param.lambda = lambda;
    param.output_sigma_factor=output_sigma_factor;
    param.interp_factor=interp_factor;
    param.cell_size=cell_size;
    param.features=features;
    param.PSRange = 0.6/(1+padding);
    param.deleteThresholdP =5; % 正样本PSR阈值
    param.deleteThresholdN =15; % 负样本PSR阈值
    param.numParticles=50;
    param.yellowArea = 1.5;
    param.blueArea = 9;

    init=1;
    continueScale =0;

	%if the target is large, lower the resolution, we don't need that much detail
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end
    
	if show_visualization,  %create video interface
		update_visualization = show_video(img_files, video_path, resize_image);
	end
		
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
	targetSize = zeros(numel(img_files), 2);  %to calculate precision
	for frame = 1:numel(img_files),
		%load image
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);% 图像缩至原来的一半
        end
        tic()     
        %initialize parts
        if frame==init
            if frame==1
                pars = repmat([pos target_sz*0.6]',[1 param.numParticles]);% 复制数组
                contexts=addParts(im,pars,pos,target_sz,param);
            end

            points=[];
            n=numel(contexts);
            for i=1:n
                if contexts{i}.target% 正样本
                    p = contexts{i}.pos;
                    points = [points; p];
                    contexts{i}.displace = pos -p;
                end
            end
            original = target_sz;

            area = prod(std(points));% std()标准差
            scale = prod(target_sz)/area;
            pnum = size(points,1);% 正样本个数
        end       
        %track the parts
        for i=1:numel(contexts)
            contexts{i} = kcftracker(im,contexts{i},param);
        end
        
        iiid=1:100;
        ixx=1:100;
        % 投票表决得到目标信息       
        if frame~=1           
            %voting
            [pos,target_sz] = voting(pos,target_sz,contexts,1);
            [contexts,changed]=resetParts(im,pos,target_sz,contexts,param);
            [pos,~] = voting(pos,target_sz,contexts,changed);
             
            n=numel(contexts);

            nn=0;
            points=[];

            target_sz = original*changed;

           if changed >1.2 || changed<0.8
               continueScale = continueScale +1;
           else
               continueScale =0;
           end
           % 连续5帧尺度变化都很大就
           if continueScale > 5
               init=frame+1;
               continueScale =0;
           end                     
           %%for draw          
            n=numel(contexts);         
            psr=[];

            for i=1:n
                if isfield(contexts{i},'psr') && contexts{i}.psr >0 
                    prob = contexts{i}.psr * size(contexts{i}.traj,1);
                    if isfield(contexts{i},'motionP') 
                        prob = prob* contexts{i}.motionP;
                    end
                    psr=[psr prob];
                    iiid=[iiid i];
                end
            end
            
            [~,ixx] = sort(psr,'descend');
            csize = 20;
            if csize > size(psr,2)
                csize = size(psr,2);
            end
            ixx = ixx(1:csize);
        end
        
        positions(frame,:) = pos;
        targetSize(frame,:) = target_sz;
		time = time + toc();
                
		%visualization
		if show_visualization,
            csize = 10;
            if csize > size(ixx,2)
                csize = size(ixx,2);
            end
            
            box = zeros(csize,5);
            for i=1:csize
                box(i,:) = [contexts{iiid(ixx(i))}.pos([2,1]) - contexts{iiid(ixx(i))}.target_sz([2,1])/2, ...
                contexts{iiid(ixx(i))}.target_sz([2,1]), contexts{iiid(ixx(i))}.target];
            end
            t=[pos([2 1]) - target_sz([2,1])/2, target_sz([2 1]), 3];
            box = [t;box];
            %加入蓝色区域
            blue=param.blueArea;
            area= prod(target_sz);
            areaS = [sqrt(area), sqrt(area)];
            t=[pos([2 1]) - blue*areaS([2,1])/2, blue*areaS([2 1]), 4];
            box = [t;box];
            %加入黄色区域
            yellew=param.yellowArea;
            t=[pos([2 1]) - yellew*target_sz([2,1])/2, yellew*target_sz([2 1]), 5];
            box = [t;box];
            stop = update_visualization(frame, box);
			if stop
                break
            end  %user pressed Esc, stop early			
			drawnow
        end      
    end
        if resize_image,
            positions = positions * 2;
            targetSize =targetSize * 2;
        end

end
