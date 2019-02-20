%   Reliable Patch Trackers: Robust Visual Tracking by Exploiting Reliable Patches
%
%   Yang Li, 2015
%   http://ihpdep.github.io
%
%   This is the research code of our RPT tracker. You can check the
%   details in our CVPR paper.

function [ contexts,changed ] = resetParts( im,pos,target_sz,contexts,param )

scalePercent = 0.3;
% ������֡��
frameRange = 5;
% �����ӵı���
positivePercent = 0.8;

%delete the unstable patchesɾ�����ɿ�������
n=numel(contexts);% ��������
index = [];% ��Ҫ���µ����ӵ�����

trajs=[]; % �켣ֵ
trajLabels=[];% ��ע�켣Ϊ����

target=[];% ���������ӵ�����ֵ

psrP=[];% �����ӵ���Ӧֵ
idP = [];% �����������ӵ�����ֵ

psrN=[];% �����ӵ���Ӧֵ
idN= [];% �����ĸ����ӵ�����ֵ

changed=1;% �߶ȱ仯ֵ
area= prod(target_sz); %Ŀ�������
areaS = [sqrt(area), sqrt(area)];% ��Ŀ��ͬ����������εĳ��Ϳ�
% ��������
for i=1:n
    % ������
    if contexts{i}.target>0        
        if (contexts{i}.psr < param.deleteThresholdP && contexts{i}.psr>0)...% �����Ŷ�
                || prod(contexts{i}.target_sz) > area ...% �߶�̫��
            || outImage(size(im),contexts{i}.pos) || ~inBox(target_sz,pos,contexts{i}.pos,param.yellowArea)% ��Ŀ��̫Զ 1.5
            index=[index i];% ��Ҫ���������ӵ�����
        else         
              if  size(contexts{i}.traj,1) >= frameRange 
                  psrP = [psrP contexts{i}.psr];% �����ӵ���Ӧֵ
                  idP = [idP i];% �����������ӵ�����ֵ
                  tmp=contexts{i}.traj(1:frameRange,:);
                  trajs=[trajs tmp(:)];
                  trajLabels = [trajLabels 1 ];% ��ע�켣Ϊ��
                  target=[target i];% ���������ӵ�����ֵ
              end
        end
    % ������
    else        
        if (contexts{i}.psr < param.deleteThresholdN && contexts{i}.psr>0)...
            || outImage(size(im),contexts{i}.pos) ...
             || prod(contexts{i}.target_sz) > area  ...
            || ~inBox(areaS,pos,contexts{i}.pos,param.blueArea)...% ����Ŀ����9������
            || inBox(target_sz,pos,contexts{i}.pos,param.yellowArea)% ��Ŀ����1.5����
            index=[index i];% ��Ҫ���������ӵ�����
        else          
             if  size(contexts{i}.traj,1) >= frameRange
                 psrN = [psrN contexts{i}.psr];% �����ӵ���Ӧֵ
                 idN = [idN i];% �����ĸ����ӵ�����ֵ
                 tmp=contexts{i}.traj(1:frameRange,:);%
                 trajs=[trajs tmp(:)];
                 trajLabels = [trajLabels 0 ];% ��ע�켣Ϊ��
                 target=[target i];
             end
        end
    end

end

psrPM=[];% �����ӵ��������Ŷ�
psrNM=[];% �����ӵ��������Ŷ�
lambda =1;% ���������켣�����Ĳ���

if size(unique(trajLabels),2) ~=1 % unique()���ص��������ж��ص�ֵ�����˵����������ӵĹ켣��ע���Ϊ1
   A = EuDist2(trajs');
   for i=1:size(target,2)% �������еı�������
       pL=(trajLabels==1);
       
       pD = A(i,pL);
       pD = sum(pD)/size(pD,2);
       
       nD = A(i,~pL);
       nD = sum(nD)/size(nD,2);
        if contexts{target(i)}.target>0 % �����ӵĹ켣����
            pMotion = nD - pD;% ���й�ʽ10
            ep=exp(lambda*pMotion);
            contexts{target(i)}.motionP=ep;
            psrPM = [psrPM ep*contexts{target(i)}.psr]; % �����ӵ��������Ŷ�
        else % �����ӵĹ켣����
            pMotion =  pD - nD;
            ep=exp(lambda*pMotion);
            contexts{target(i)}.motionP=ep;
            psrNM = [psrNM ep*contexts{target(i)}.psr];  % �����ӵ��������Ŷ�
        end
   end
   psrN = psrN .* psrNM;% �����ʽ�Ͱ�������Ӧǿ�ȹ�ʽ�е�ָ��2
   psrP = psrP.*psrPM;
end
    %���������ĵ���
    if  size(idP,2) > round(n*positivePercent)% �ж������ӵı���
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
% ������Ҫ�ز���������
psr =[];
pars = zeros(4,1);
k=1;
positive = [];
psrP=[];
points=[];
for i=1:n
    if sum(index==i) ==0 && contexts{i}.psr >0%û����Ҫ�ز���������
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
% ����������ͶƱ
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
    %ȡ��ֵΪ�߶ȵı仯ֵ
    changed = s(round(size(s,2)/2)+1);
end      
% �ز���        
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
% �ж�����λ���Ƿ��Ѿ�����ͼ������
function b=outImage(WinSize,p)

    if min(p < WinSize - [1 1]) * min(p > [2 2]) > 0
        b=false;
    else
        b=true;
    end
end

