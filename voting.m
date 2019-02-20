%   Reliable Patch Trackers: Robust Visual Tracking by Exploiting Reliable Patches
%
%   Yang Li, 2015
%   http://ihpdep.github.io
%
%   This is the research code of our RPT tracker. You can check the
%   details in our CVPR paper.
%   投票决定目标位置和尺度

function  [position,target_sz] = voting(pos,target_sz,contexts,scale)
    % Detailed explanation goes here
    n=numel(contexts);
    position = pos;
    pos=[0,0];
    nn=0;
    points=[];
    psr=[];
    % 只有正样本参与投票
    for i=1:n
        if contexts{i}.target && isfield(contexts{i},'psr') && contexts{i}.psr >0 % 是正样本，且响应大于0
            p=contexts{i}.pos; % 粒子位置
            tmp =  p+ contexts{i}.displace*scale; % 每个粒子向目标的投票值
            points=[points; tmp];
            prob = contexts{i}.psr * size(contexts{i}.traj,1);
            if isfield(contexts{i},'motionP') 
                prob = prob* contexts{i}.motionP;
            end
            psr=[psr prob];
            pos = tmp +pos;
            nn=nn+1;
        end
    end
    if nn~=0
        psr=psr./sum(psr);% 归一化权重
        position = psr*points;
    end   
    if min(~isnan(position)) == 0
        position=[0 0];
    end
    
end
