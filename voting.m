%   Reliable Patch Trackers: Robust Visual Tracking by Exploiting Reliable Patches
%
%   Yang Li, 2015
%   http://ihpdep.github.io
%
%   This is the research code of our RPT tracker. You can check the
%   details in our CVPR paper.
%   ͶƱ����Ŀ��λ�úͳ߶�

function  [position,target_sz] = voting(pos,target_sz,contexts,scale)
    % Detailed explanation goes here
    n=numel(contexts);
    position = pos;
    pos=[0,0];
    nn=0;
    points=[];
    psr=[];
    % ֻ������������ͶƱ
    for i=1:n
        if contexts{i}.target && isfield(contexts{i},'psr') && contexts{i}.psr >0 % ��������������Ӧ����0
            p=contexts{i}.pos; % ����λ��
            tmp =  p+ contexts{i}.displace*scale; % ÿ��������Ŀ���ͶƱֵ
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
        psr=psr./sum(psr);% ��һ��Ȩ��
        position = psr*points;
    end   
    if min(~isnan(position)) == 0
        position=[0 0];
    end
    
end
