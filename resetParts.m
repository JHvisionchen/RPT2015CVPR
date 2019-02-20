%   Reliable Patch Trackers: Robust Visual Tracking by Exploiting Reliable Patches
%
%   Yang Li, 2015
%   http://ihpdep.github.io
%
%   This is the research code of our RPT tracker. You can check the
%   details in our CVPR paper.

function [ contexts,changed ] = resetParts( im,pos,target_sz,contexts,param )

scalePercent = 0.3;
% 连续的帧数
frameRange = 5;
% 正粒子的比例
positivePercent = 0.8;

%delete the unstable patches删除不可靠的粒子
n=numel(contexts);% 粒子数量
index = [];% 需要更新的粒子的索引

trajs=[]; % 轨迹值
trajLabels=[];% 标注轨迹为正或负

target=[];% 保留正粒子的索引值

psrP=[];% 正粒子的响应值
idP = [];% 保留的正粒子的索引值

psrN=[];% 负粒子的响应值
idN= [];% 保留的负粒子的索引值

changed=1;% 尺度变化值
area= prod(target_sz); %目标框的面积
areaS = [sqrt(area), sqrt(area)];% 与目标同面积的正方形的长和宽
% 所有样本
for i=1:n
    % 正样本
    if contexts{i}.target>0        
        if (contexts{i}.psr < param.deleteThresholdP && contexts{i}.psr>0)...% 低置信度
                || prod(contexts{i}.target_sz) > area ...% 尺度太大
            || outImage(size(im),contexts{i}.pos) || ~inBox(target_sz,pos,contexts{i}.pos,param.yellowArea)% 离目标太远 1.5
            index=[index i];% 需要丢弃的粒子的索引
        else         
              if  size(contexts{i}.traj,1) >= frameRange 
                  psrP = [psrP contexts{i}.psr];% 正粒子的响应值
                  idP = [idP i];% 保留的正粒子的索引值
                  tmp=contexts{i}.traj(1:frameRange,:);
                  trajs=[trajs tmp(:)];
                  trajLabels = [trajLabels 1 ];% 标注轨迹为正
                  target=[target i];% 保留正粒子的索引值
              end
        end
    % 负样本
    else        
        if (contexts{i}.psr < param.deleteThresholdN && contexts{i}.psr>0)...
            || outImage(size(im),contexts{i}.pos) ...
             || prod(contexts{i}.target_sz) > area  ...
            || ~inBox(areaS,pos,contexts{i}.pos,param.blueArea)...% 超出目标框的9倍距离
            || inBox(target_sz,pos,contexts{i}.pos,param.yellowArea)% 在目标框的1.5倍内
            index=[index i];% 需要丢弃的粒子的索引
        else          
             if  size(contexts{i}.traj,1) >= frameRange
                 psrN = [psrN contexts{i}.psr];% 负粒子的响应值
                 idN = [idN i];% 保留的负粒子的索引值
                 tmp=contexts{i}.traj(1:frameRange,:);%
                 trajs=[trajs tmp(:)];
                 trajLabels = [trajLabels 0 ];% 标注轨迹为负
                 target=[target i];
             end
        end
    end

end

psrPM=[];% 正粒子的整体置信度
psrNM=[];% 负粒子的整体置信度
lambda =1;% 文中描述轨迹分析的参数

if size(unique(trajLabels),2) ~=1 % unique()返回的是数组中独特的值，这段说的是如果粒子的轨迹标注类别不为1
   A = EuDist2(trajs');
   for i=1:size(target,2)% 对于所有的保留粒子
       pL=(trajLabels==1);
       
       pD = A(i,pL);
       pD = sum(pD)/size(pD,2);
       
       nD = A(i,~pL);
       nD = sum(nD)/size(nD,2);
        if contexts{target(i)}.target>0 % 正粒子的轨迹分析
            pMotion = nD - pD;% 文中公式10
            ep=exp(lambda*pMotion);
            contexts{target(i)}.motionP=ep;
            psrPM = [psrPM ep*contexts{target(i)}.psr]; % 正粒子的整体置信度
        else % 负粒子的轨迹分析
            pMotion =  pD - nD;
            ep=exp(lambda*pMotion);
            contexts{target(i)}.motionP=ep;
            psrNM = [psrNM ep*contexts{target(i)}.psr];  % 负粒子的整体置信度
        end
   end
   psrN = psrN .* psrNM;% 这个公式就包含了响应强度公式中的指数2
   psrP = psrP.*psrPM;
end
    %正负比例的调整
    if  size(idP,2) > round(n*positivePercent)% 判断正粒子的比例
         [~,iid] =sort(psrP);
        for i=1:size(idP,2) - round(n*positivePercent)
            index=[index idP(iid(i))];
        end
    else
        if  size(idN,2) > round(n*(1-positivePercent))
             [~,iid] =sort(psrN);
            for i=1:size(idN,2) - round(n*(1-positivePercent))
                index=[index idN(iid(i))];
            end
        end
    end
% 处理需要重采样的粒子
psr =[];
pars = zeros(4,1);
k=1;
positive = [];
psrP=[];
points=[];
for i=1:n
    if sum(index==i) ==0 && contexts{i}.psr >0%没有需要重采样的粒子
            ppp=contexts{i}.psr * size(contexts{i}.traj,1);
            psr = [psr ppp];
            pars(1:2,k) = contexts{i}.pos;
            pars(3:4,k) = contexts{i}.target_sz;
            k=k+1;
            if contexts{i}.target >0 &&   isfield(contexts{i},'motionP') && contexts{i}.motionP >1
                positive=[positive i];
            end
    end
end
% 正样本参与投票
if size(positive,2)>scalePercent*n
    scale =[];
    for i=1:size(positive,2)
        for j=i+1:size(positive,2)
            a = sqrt(sum((contexts{positive(i)}.pos - contexts{positive(j)}.pos).^2));
            r =  sqrt(sum((contexts{positive(i)}.displace - contexts{positive(j)}.displace).^2));
            scale = [scale a/r];
        end
    end
    [s, ~] = sort(scale);
    %取中值为尺度的变化值
    changed = s(round(size(s,2)/2)+1);
end      
% 重采样        
if ~isempty(index)
    if isempty(psr)
        psr=1;
        pars(1:2,1) = pos;
        pars(3:4,1) = 0.6*target_sz;
    end
    k=size(index,2);
    psr = psr./sum(psr);
    cumconf = cumsum(psr);
    a=repmat(rand(1,size(cumconf,2)),[k,1]);
    b=repmat(cumconf',[1,k]);
    idx = floor(sum( a'> b,1))+1;
    pars = pars(:,idx);

    addContexts=addParts(im,pars,pos,target_sz,param);

    for i=1:size(index,2)
        contexts{index(i)} = addContexts{i};
    end
end
    
end
% 判断粒子位置是否已经出了图像区域
function b=outImage(WinSize,p)

    if min(p < WinSize - [1 1]) * min(p > [2 2]) > 0
        b=false;
    else
        b=true;
    end
end

