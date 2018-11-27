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
        Coord{i} = [Coord{i} Edges{j}];
    end
    Trans{i} = Act_array;
    Coord{i} = unique(Coord{i});
end


%% Aircraft - agents
agents = 2;

s = cell(agents,1);
s0 = s;
A = s;
A{1} = [1 6];
A{2} = [8 5];
Rew_Agent = s;
Q = s;
Path = Q;

%% Offline
for q = 1:agents
   s0{q} = A{q}(1);
   Rew_Agent{q} = Rew;
   Rew_Agent{q}(:,A{q}(2)) = Rew_Agent{q}(:,A{q}(2)) - 10;
   Q{q} = MDP_VI(Trans,Rew_Agent{q},A{q}(1),A{q}(2));
end

%% Online
looper = 1;
s = s0;
hist = cell(agents,1);
time = 1;
for q = 1:agents;hist{q}=s{q};end
while looper~= 0
    a = Coordinate(Q,s,Coord,1);
    time = time+1;
    for q = 1:agents
        s{q} = randsample(1:n,1,true,Trans{s{q}}(a{q},:));
        hist{q,time} = s{q};
    end
    looper = 1;
    for q = 1:agents
        looper = looper && s{q}==A{q}(2);
    end
end


%% For Plotting
s = s0;
for q = 1:agents
    s{q} = A{q}(1);
    Path{q} = s{q};
    while s{q}~=A{q}(2)
        [~,a] = min(Q{q}{s{q}});
        s{q} = Edges{s{q}}(a);
        Path{q} = [Path{q} s{q}];
    end
end


plot(States(:,1),States(:,2),'*')
axis([-1 5 -1 5])
for i = 1:size(States,1)
   circle(States(i,1),States(i,2),1);
end

h_path1 = plot_path(States(Path{1},:),'r');
h_path2 = plot_path(States(Path{2},:),'b');
hold on
h_agent1 = plot(States(A{1}(1),1),States(A{1}(1),2),'rs','MarkerFaceColor','r');
h_agent2 = plot(States(A{2}(1),1),States(A{2}(1),2),'bs','MarkerFaceColor','b');
hold off


%
looper = 1;
s = s0;
while looper~= 0
    
    set(h_agent1,'XData',States(s{1},1),'YData',States(s{1},2));
    set(h_agent2,'XData',States(s{2},1),'YData',States(s{2},2));
    drawnow
    looper = 1;
    for q = 1:agents
        looper = looper && s{q}==A{q}(2);
    end
end


figure;
axis([-1 5 -1 5])
hold on;
for i = 1:n
    for j = 1:length(Edges{i})
       plot_edge(States(i,1),States(i,2),States(j,1),States(j,2));
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
    gamma = 0.9;
    epsilon = 0.25;
    while sum(abs(cell2mat(cellfun(@minus,Q1,Q,'Un',0)))) > 1e-3*n
        s = s0;
        Q1 = Q;
        while ismember(s,Goal) ~= 1
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

function a = Coordinate(Q,s,Coord,flag)
    % Co-ordinate switch - flag
    agents = size(Q,1);
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
            co_ord_graph{q} = agent_idx(ismember(air_loc,Coord{q}));
        end
        for q = 1:agents
            if isempty(co_ord_graph{q})
                [~,a{q}] = min(Q{q}{s{q}});
            else
                
            end
        end
    end
end