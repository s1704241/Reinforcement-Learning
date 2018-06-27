
%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;

%% PROBLEM SPECIFICATION:

blockSize = 5 ; % This will function as the dimension of the road basis 
% images (blockSize x blockSize), as well as the view range, in rows of
% your car (including the current row).

n_MiniMapBlocksPerMap = 5 ; % determines the size of the test instance. 
% Test instances are essentially road bases stacked one on top of the
% other.

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

%discountFactor_gamma = 1 ; % if needed

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. In order: paved 
% square, non-paved square, and car collision. Agents can occupy the same
% square as another car, and the collision does not end the instance, but
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly 
% transitions into one of the above 3 states. If one of those states would 
% be outside the map, the next state will be the one above the current one.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid 
% maps, complete with an initial location for your agent. (Also see the 
% GridMap class).

tempGrid = [ roadBasisGridMaps(2).Grid; ...
  roadBasisGridMaps(3).Grid; roadBasisGridMaps(2).Grid; ...
  roadBasisGridMaps(8).Grid; roadBasisGridMaps(7).Grid ] ;

tempStart = [ n_MiniMapBlocksPerMap * blockSize, 1 ] ;

tempMarkerRescaleFactor = 1/( (25^2)/36 ) ;

MDP_1 = GridMap(tempGrid, tempStart, tempMarkerRescaleFactor, ...
    probabilityOfUniformlyRandomDirectionTaken) ;

% Appending a matrix (same size size as the grid) with the locations of 
% cars:
MDP_1.CarLocations = [0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     1     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     1     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     1     0     0     0 ; ...
                      0     0     1     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     1     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     0     0     0 ; ...
                      0     0     1     0     0 ; ...
                      0     0     0     0     0 ];


% Appending the reward function (depends on next state and, only for 
% terminal states, on the current state):
MDP_1.RewardFunction = generateRewardFunction( MDP_1, rewards ) ;


%% Deterministic Policy to evaluate:
pi_test1 = UP * ones( MDP_1.GridSize ); % Default action: up.
pi_test1(:, 1) = UP_RIGHT; % When on the leftmost column, go up right.
pi_test1(:, 5) = UP_LEFT ; % When on the rightmost column, go up left.
pi_test1(:, 3) = UP_LEFT ; % When on the center column, go up left.

pi_test1_stateNumbers = zeros(1,125);
pi_test1_stateNumbers(:) = pi_test1';

%%
currentTimeStep = 0 ;
currentMap = MDP_1 ;
agentLocation = currentMap.Start ;
startingLocation = agentLocation ; % Keeping record of initial location.

% If you need to keep track of agent movement history:
%
agentMovementHistory = zeros(episodeLength+1, 2) ;
%
agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;


%% PRINT MAP:
% You can update viewableGridMap in a similar way as below, in order to
% keep track of the current visible area for your car (don't use this with
% road bases since the whole map should be visible at any time in that case
% ): 
viewableGridMap = ...
    setCurrentViewableGridMap( MDP_1, agentLocation, blockSize ) ;
% When printing $viewableGridMap.Grid$ notice that the row numbers no
% longer correspond to the original test map rows. Use $agentLocation(1)$  
% to keep track of your current row in the complete test map.

refreshScreen % See $refreshScreen$ function for details.


%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
% Simulating agent behaviour when following the policy defined by 
% $pi_test1$.
%
% Commented lines also have examples of use for $GridMap$'s $getReward$ and
% $getTransitions$ functions, which act as our reward and transition
% functions respectively.

realAgentLocation = agentLocation ; % The location on the full test map.
Return = 0;

V_former = zeros(MDP_1.GridSize);
V_later = zeros(MDP_1.GridSize);
new_map = pi_test1;
last_map = new_map;
current_map = zeros(MDP_1.GridSize);
a1 = 0;
a2 = 0;
a3 = 0;

while true
    V_later = zeros(MDP_1.GridSize);
    for row = 2:size(V_later, 1)
        for col = 1:size(V_later,2)
            [state,prob] = MDP_1.getTransitions([row,col],last_map(row,col));
            for x = 1:size(state,1)
                reward = MDP_1.getReward([row,col], state(x,:), last_map(row,col));
                V_later(row,col) = V_later(row,col) + prob(x) * ( reward + V_former(state(x,1), state(x,2)) );
            end
        end
    end
    V_former = V_later;
    for r = 2:size(V_later, 1)
        for c = 1:size(V_later,2)
            Q_a1=0;
            Q_a2=0;
            Q_a3=0;
            [state,prob] = MDP_1.getTransitions([r,c],1);
            for x = 1:size(state,1)
                reward = MDP_1.getReward([r,c], state(x,:), 1);
                Q_a1 = Q_a1 + prob(x) * ( reward + V_later(state(x,1), state(x,2)) );
            end
            [state,prob] = MDP_1.getTransitions([r,c],2);
            for x = 1:size(state,1)
                reward = MDP_1.getReward([r,c], state(x,:), 2);
                Q_a2 = Q_a2 + prob(x) * ( reward + V_later(state(x,1), state(x,2)) );
            end
            [state,prob] = MDP_1.getTransitions([r,c],3);
            for x = 1:size(state,1)
                reward = MDP_1.getReward([r,c], state(x,:), 3);
                Q_a3 = Q_a3 + prob(x) * ( reward + V_later(state(x,1), state(x,2)) );
            end
            [maxQ, maxQ_index]=max([Q_a1, Q_a2, Q_a3]);     
            new_map(r,c) = maxQ_index;
        end
    end
    
    if last_map == new_map;
        break;
    else
        last_map = new_map;
    end
    %last_map = new_map;
%     for r = 3:25
%         if V_later(r-1,1) > V_later(r-1,2)
%         	new_map(r,1) = 2;
%         else
%             new_map(r,1) = 3;
%         end
% 
%         if V_later(r-1,4) > V_later(r-1,5)
%             new_map(r,5) = 1;
%         else
%             new_map(r,5) = 2;
%         end
%     
%     
%         for c = 2:4
%             if max(V_later(r-1,c-1:c+1)) == V_later(r-1,c-1);
%                 new_map(r,c) = 1;
%             elseif max(V_later(r-1,c-1:c+1)) == V_later(r-1,c);
%                 new_map(r,c) = 2;
%             else
%                 new_map(r,c) = 3;
%             end
%         end
%     end
%     if current_map == new_map;
%         break;
%     else
%         current_map = new_map;
%     end
end
    
    
  
for i = 1:episodeLength
    
    actionTaken = new_map( realAgentLocation(1), realAgentLocation(2) );
    
    % The $GridMap$ functions $getTransitions$ and $getReward$ act as the 
    % problems transition and reward function respectively.
    %
    % $actionMoveAgent$ can be used to simulate agent (the car) behaviour.
    
%     [ possibleTransitions, probabilityForEachTransition ] = ...
%         MDP_1.getTransitions( realAgentLocation, actionTaken );
%     [ numberOfPossibleNextStates, ~ ] = size(possibleTransitions);
%     previousAgentLocation = realAgentLocation;
    
    [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
        agentMovementHistory ] = ...
        actionMoveAgent( actionTaken, realAgentLocation, MDP_1, ...
        currentTimeStep, agentMovementHistory, ...
        probabilityOfUniformlyRandomDirectionTaken ) ;

%     MDP_1.getReward( ...
%             previousAgentLocation, realAgentLocation, actionTaken )
    
    Return = Return + agentRewardSignal;
    
    % If you want to view the agents behaviour sequentially, and with a 
    % moving view window, try using $pause(n)$ to pause the screen for $n$
    % seconds between each draw:
       
    [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
        MDP_1, realAgentLocation, blockSize );
    % $agentLocation$ is the location on the viewable grid map for the 
    % simulation. It is used by $refreshScreen$.
    
    currentMap = viewableGridMap ; %#ok<NASGU>
    % $currentMap$ is keeping track of which part of the full test map
    % should be printed by $refreshScreen$ or $printAgentTrajectory$.
    
    refreshScreen
    
    pause(0.15)
    
end

currentMap = MDP_1 ;
agentLocation = realAgentLocation ;

Return

printAgentTrajectory



