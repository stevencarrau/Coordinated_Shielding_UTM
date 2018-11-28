close all
% Grid states
rng(1)
n = 8;
radius = 2;
%States = rand(n,2)*4;

%% Environment
States = [0 0;
          1 1;
          1 2.8;
          2 2;
          1.3 -0.2;
          4 3;
          2.8 3.4;
          3.1 1.8
          ];
Edges = cell(n,1);
Rew = ones(n)*99;

for i = 1:size(States,1)
    Edges{i} = [];
    for j = 1:size(States,1)
        if norm(States(i,:)-States(j,:))<radius
           Edges{i} = [Edges{i} j];
           Rew(i,j) = norm(States(i,:)-States(j,:));
           if i==j
               Rew(i,j) = 1.5;
           end
        end
    end
end

Trans = cell(n,1);
Coord = cell(n,1);

for i = 1:n
    m = size(Edges{i,1},2);
    Act_array = zeros(m,n);
    Coord{i} = [Edges{i}];
    for j = 1:m
        if i == Edges{i}(j)
            Act_array(j,Edges{i}(j)) = 1;
        else
            Act_array(j,Edges{i}(j)) = 0.9;
            Act_array(j,i) = 0.1;
        end
        Coord{i} = unique([Coord{i} Edges{Edges{i}(j)}]);
    end
    Trans{i} = Act_array;
end


%% Aircraft - agents
agents = 2;

s = cell(agents,1);
s0 = s;
T = s;
A = s;
A{1} = [1 6];
A{2} = [8 5];
Rew_Agent = s;
Q = s;
Path = Q;

%% Offline
for q = 1:agents
   s0{q} = A{q}(1);
   T{q} = A{q}(2);
   Rew_Agent{q} = Rew;
   Rew_Agent{q}(:,T{q}) = Rew_Agent{q}(:,T{q}) - 100;
   Q{q} = MDP_VI(Trans,Rew_Agent{q},A{q}(1),A{q}(2));
   disp("Calculated route")
end
% load('PreCalcQ.mat');

%% Online
s = s0;
hist = cell(agents,1);
hist_sts = cell(1,1);
hist_Path = cell(agents,1);
time = 1;
for q = 1:agents
    Path{q}=s{q};
    hist{q}=s{q};
    hist_sts{1} = [];
    hist_Path{q} = [s{q} Path2Go(q,T{q},Q,Edges,s{q})];
end
while isequal(s,T)~=1
    [a,sts] = Coordinate(Q,s,Coord,Edges,1);
    time = time+1;
    for q = 1:agents
        s{q} = randsample(1:n,1,true,Trans{s{q}}(a{q},:));
        Path{q} = [Path{q} s{q}];
        hist{q,time} = s{q};
        hist_sts{1,time} = sts;
        hist_Path{q,time} = [Path{q} Path2Go(q,T{q},Q,Edges,s{q})];
    end
end


%% For Plotting

plot(States(:,1),States(:,2),'*')
axis([-1 5 -1 5])
for i = 1:size(States,1)
   h_circ{i} = circle(States(i,1),States(i,2),1);
end

h_path1 = plot_path(States(hist_Path{1,1},:),'r');
h_path2 = plot_path(States(hist_Path{2,1},:),'b');
hold on
h_agent1 = plot(States(A{1}(1),1),States(A{1}(1),2),'rs','MarkerFaceColor','r','MarkerSize',18);
h_agent2 = plot(States(A{2}(1),1),States(A{2}(1),2),'bs','MarkerFaceColor','b','MarkerSize',18);
hold off


%
F(1) = getframe(gcf);
for i = 1:size(hist,2)
    for z = 1:n
        set(h_circ{n},'FaceColor','g');
    end
    if isempty(hist_sts{i})~=1
        for k = hist_sts{i}   
            set(h_circ{k},'FaceColor','r');
        end
    end
    drawnow
    set(h_agent1,'XData',States(hist{1,i},1),'YData',States(hist{1,i},2));
    set(h_agent2,'XData',States(hist{2,i},1),'YData',States(hist{2,i},2));
    set(h_path1,'XData',States(hist_Path{1,i},1),'YData',States(hist_Path{1,i},2))
    set(h_path2,'XData',States(hist_Path{2,i},1),'YData',States(hist_Path{2,i},2))
    drawnow
    
    F(i) = getframe(gcf);
end
writerObj = VideoWriter('2Agents.avi');
writerObj.FrameRate = 0.2;

figure;
axis([-1 5 -1 5])
hold on;
for i = 1:n
    for j = 1:length(Edges{i})
       plot_edge(States(i,1),States(i,2),States(j,1),States(j,2));
    end
end


function Path = Path2Go(agent,T,Q,Edges,s)
    s1 = s;
    Path = [];
    while s1~=T
      [~,a] = min(Q{agent}{s1});
      s1 = Edges{s1}(a);
      Path = [Path s1];
    end
end


function h = circle(x,y,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = fill(xunit, yunit,'g');
set(h,'facealpha',0.2);
hold off
end

function h = plot_path(x,col)
    hold on 
    h = plot(x(:,1),x(:,2),'*-');
    set(h,'Color',col);
    set(h,'MarkerEdgeColor',col);
    hold off
end

function h2 = plot_edge(x1,y1,x2,y2)
    hold on
    h2 = plot([x1,x2],[y1,y2],'k-x');
    set(h2,'MarkerEdgeColor','r');
    hold off
end

function [Q] = MDP_VI(Trans,Rew,s0,Goal)
    n = size(Rew,1);
    Q = cell(n,1);
    Q1 = Q;
    for i = 1:n
       m = size(Trans{i},1); % number of actions
       Q{i} = 20*ones(m,1);
       Q1{i} = zeros(m,1);
    end
    cnt = 1;
    
    delta = 1;
    alpha = 0.5;
    gamma = 0.95;
    epsilon = 0.4;
    while sum(abs(cell2mat(cellfun(@minus,Q1,Q,'Un',0)))) > 1e-5*n
        s = s0;
        Q1 = Q;
%         while ismember(s,Goal) ~= 1
        for k_z = 1:15
            if rand(1)<epsilon
                a_t = randsample(1:size(Trans{s},1),1);
            else
                [~,a_t] = min(Q{s});
            end
            delta = 0;
            for k = 1:n
               delta = delta + Trans{s}(a_t,k)*(Rew(s,k)+gamma*(1-epsilon)*min(Q{k})+epsilon/4*sum(Q{k}));
            end
            Q{s}(a_t) = Q{s}(a_t)+alpha/cnt*(delta-Q{s}(a_t));
            s = randsample(1:n,1,true,Trans{s}(a_t,:));
        end
        cnt = cnt+1;
    end
    
end

function [a,sts] = Coordinate(Q,s,Coord,Edges,flag)
    % Co-ordinate switch - flag
    agents = size(Q,1);
    sts = [];
    a = cell(agents,1);
    co_ord_graph = a;
    n = size(s,1);
    if flag == 0 % No co-ordination at all
        for q = 1:agents
            [~,a{q}] = min(Q{q}{s{q}});
        end
    else
        for q = 1:agents
            air_loc = cell2mat(s);
            air_loc(q) = -1; %Don't need to co-ordinate with ourselves
            agent_idx = 1:agents;
            co_ord_graph{q} = agent_idx(ismember(air_loc,Coord{s{q}}));
        end
        graph_space = unique(cell2mat(co_ord_graph'));
        if isempty(graph_space)
            % No co-ordination required
            for q = 1:agents
                [~,a{q}] = min(Q{q}{s{q}});
            end
        else
            % Co-ordinated
            % Assign leftovers (not co-ordinating with the rest)
            leftovers = 1:n;
            leftovers(ismember(leftovers,graph_space)) = [];
            for w = leftovers
               [~,a{w}] = min(Q{w}{s{w}}); 
            end
            
            % DBN framework
            new_Q = cell(size(graph_space,2),1);
            new_s = new_Q;
            new_graph = new_Q;
            for k = 1:length(graph_space)
               new_Q{k} = Q{graph_space(k)}{s{graph_space(k)}};
               new_s{k} = s{graph_space(k)};
               new_graph{k} = Coord{graph_space(k)};
            end
            [a_set,sts] = min_DBN(new_Q,new_s,new_graph,Edges);
            for k = 1:length(graph_space)
                a{graph_space(k)} = a_set(k);
            end
        end
    end
end

function [a,sts] = min_DBN(Q,s,graph,Edges)
    agents = size(s,1);
    sts = [];
    % Only 2 or 3 agent co-ordination for now
    if agents==2
        a =zeros(2,1);
       [Qx,Qy] = meshgrid(Q{1},Q{2}); 
        comp_Q = Qx+Qy;
        for q = 1:agents
            % Remove occupied operators
            air_loc = cell2mat(s');
            air_loc(q) = -1;
            [~,act_idx] = ismember(air_loc,Edges{s{q}});
            if q==1
                comp_Q(:,act_idx(act_idx~=0)) = comp_Q(:,act_idx(act_idx~=0)) + 100;
            elseif q==2
                comp_Q(act_idx(act_idx~=0),:) = comp_Q(act_idx(act_idx~=0),:) + 100;
            end
            % Find colliding actions
            Ext_Edges = Edges{s{q}};
            Ext_Edges(Ext_Edges==s{q})=-1; 
            for m = 1:length(Ext_Edges)
                rem_agents = 1:agents;
                rem_agents(q) = [];
                for q_z = rem_agents
                    if ismember(Ext_Edges(m),Edges{s{q_z}})
                        [~,ind_i] = ismember(Ext_Edges(m),Edges{s{q_z}});
                        if q==1
                            comp_Q(ind_i,m) = comp_Q(ind_i,m) + 100;
                        elseif q==2
                            comp_Q(m,ind_i) = comp_Q(m,ind_i) + 100;
                        end
                        sts = [sts Ext_Edges(m)];
                    end
                end
            end
        end
        [~,min_idx] = min(comp_Q(:));
        [a(2),a(1)] = ind2sub(size(comp_Q),min_idx);
    elseif agents==3
         a =zeros(3,1);
        [Qx,Qy,Qz] = meshgrid(Q{1},Q{2},Q{3});
        comp_Q = Qx+Qy+Qz;
        for q = 1:agents
            % Find colliding actions
            Ext_Edges = Edges{s{q}};
            Ext_Edges(Ext_Edges==s{q})=-1; 
            for m = 1:length(Ext_Edges)
                rem_agents = 1:agents;
                rem_agents(q) = [];
                for q_z = rem_agents
                    [~,ind_i] = ismember(Ext_Edges(m),Edges{s{q_z}});
                    if q==1 && q_z == 2
                        comp_Q(ind_i,m,:) = comp_Q(ind_i,m,:) + 100;
                    elseif q==1 && q_z == 3
                        comp_Q(ind_i,:,m) = comp_Q(ind_i,:,m) + 100;
                    elseif q==2 && q_z == 3
                        comp_Q(:,ind_i,m) = comp_Q(:,ind_i,m) + 100;
                    end
                end
            end
        end
        [~,min_idx] = min(comp_Q(:));
        [a(2),a(1),a(3)] = ind2sub(size(comp_Q),min_idx);
    else
        error("Oofta");    
    end
    sts = unique(sts);

end