
%%FEATURES : 0 for Original, 1 for Custom.
FEATURES = 0;


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

noCarOnRowProbability = 0.8 ; % the probability that there is no car 
% spawned for each row

seed = 1234;
rng(seed); % setting the seed for the random nunber generator

% Call this whenever starting a new episode:
MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, blockSize, ...
    noCarOnRowProbability, probabilityOfUniformlyRandomDirectionTaken, ...
    rewards );





%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
% Simulating agent behaviour when following the policy defined by 
% $pi_test1$.
%
% Commented lines also have examples of use for $GridMap$'s $getReward$ and
% $getTransitions$ functions, which act as our reward and transition
% functions respectively.
epsilon_greedy = 0.09;

if FEATURES == 0;
   %% Initialising the state observation (state features) and setting up the 
   % exercise approximate Q-function:
   stateFeatures = ones( 4, 5 );
   action_values = zeros(1, 3);
   next_action_values = zeros(1, 3);

   discountFactor_gamma = 0.9

   Q_test1 = zeros(4, 5, 3);
  
   %Initialising the weights for function approximation
   weight = zeros(4,5,3)

   for episode = 1:50000

        %%
        currentTimeStep = 0 ;
        MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
            blockSize, noCarOnRowProbability, ...
            probabilityOfUniformlyRandomDirectionTaken, rewards );
        currentMap = MDP ;
        agentLocation = currentMap.Start ;
        startingLocation = agentLocation ; % Keeping record of initial location.

        % If you need to keep track of agent movement history:
        agentMovementHistory = zeros(episodeLength+1, 2) ;
        agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;

        realAgentLocation = agentLocation ; % The location on the full test map.
        Return = 0;

        for i = 1:episodeLength

            % Use the $getStateFeatures$ function as below, in order to get the
            % feature description of a state:
            if i == 1
                stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
            end

            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values);



            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;

            next_stateFeatures = MDP.getStateFeatures(realAgentLocation);
            for action = 1:3
                next_action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* next_stateFeatures ) );
            end % for each possible action
            [~, next_actionTaken] = max(next_action_values);
            
               
            %epsilon_greedy
            random = rand();
            candidate = [1,2,3];
            Locate=find(next_actionTaken); 
            candidate(Locate)=[];  
            
            %when the random value epsilon_greedy is less than the threshold,
            %do the random action
            if random < 2 * epsilon_greedy / 3 
                prob = rand();
                if prob >=0.5
                    pick = 1;
                else
                    pick = 2;
                end
                next_actionTaken = candidate(pick);
                %If one of those states would be outside the map, 
                %the next state will be the one above the current one
                if (realAgentLocation(2) == 1 && next_actionTaken == 1) || ...
                        (realAgentLocation(2) == 5 && next_actionTaken == 3)
                        next_actionTaken = 2;
                end
            end
     


            prediction =  sum ( sum( weight(:,:,actionTaken) .* stateFeatures ) );
            target = agentRewardSignal + discountFactor_gamma * sum ( sum( weight(:,:,next_actionTaken) .* next_stateFeatures ) );
            
            %Use changing learning rate
            learning_rate = 0.001/episode;
            delta_w = learning_rate * (target - prediction) * stateFeatures;

            
            
            weight(:,:,actionTaken) = weight(:,:,actionTaken) + delta_w;
            Q_test1 = weight
            
            %update stateFeatures for the next state
            stateFeatures = next_stateFeatures;

        end

        currentMap = MDP ;
        agentLocation = realAgentLocation ;

    end % for each episode
    
    for episode = 1:100
    
        %%
        currentTimeStep = 0 ;
        MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
            blockSize, noCarOnRowProbability, ...
            probabilityOfUniformlyRandomDirectionTaken, rewards );
        currentMap = MDP ;
        agentLocation = currentMap.Start ;
        startingLocation = agentLocation ; % Keeping record of initial location.
        realAgentLocation = agentLocation ; % The location on the full test map.
        Return = 0;
        for i = 1:episodeLength
            stateFeatures = MDP.getStateFeatures(realAgentLocation) % dimensions are 4rows x 5columns

            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* sstateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values)

            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;
            Return = Return + agentRewardSignal;
            [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
                MDP, realAgentLocation, blockSize );
            currentMap = viewableGridMap ; %#ok<NASGU>
            refreshScreen
            pause(0.15)
        end
        currentMap = MDP ;
        agentLocation = realAgentLocation ;
        printAgentTrajectory
        pause(1)

    end % for each episode
end




if FEATURES == 1;
    %% Initialising the state observation (state features) and setting up the 
    % exercise approximate Q-function:
    stateFeatures = zeros(1, 3);
    action_values = zeros(1, 3);
    next_action_values = zeros(1, 3);
    new_stateFeatures = zeros(1, 3);
    newNext_stateFeatures = zeros(1, 3);
    M = 0;
    S = 0;

    discountFactor_gamma = 0.9;
    filter = ones(4,3);  %convolutional filter

    Q_test1 = zeros(1, 3, 3);

    %Initialising the weights for function approximation
    weight = zeros(1, 3, 3);
    for episode = 1:500        %%
        currentTimeStep = 0 ;
        MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
            blockSize, noCarOnRowProbability, ...
            probabilityOfUniformlyRandomDirectionTaken, rewards );
        currentMap = MDP ;
        agentLocation = currentMap.Start ;
        startingLocation = agentLocation ; % Keeping record of initial location.

        realAgentLocation = agentLocation ; % The location on the full test map.
        Return = 0;

        for i = 1:episodeLength

            % Use the $getStateFeatures$ function as below, in order to get the
            % feature description of a state:
            if i == 1
                stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns
                if realAgentLocation(2) == 1
                    new_stateFeatures(1,1) = 0;
                    new_stateFeatures(1,2) = stateFeatures(4,realAgentLocation(2));
                    new_stateFeatures(1,3) = stateFeatures(4,realAgentLocation(2)+1);
                elseif realAgentLocation(2) == 5
                    new_stateFeatures(1,1) = stateFeatures(4,realAgentLocation(2)-1);
                    new_stateFeatures(1,2) = stateFeatures(4,realAgentLocation(2));
                    new_stateFeatures(1,3) = 0;
                else
                    new_stateFeatures(1,1) = stateFeatures(4,realAgentLocation(2)-1);
                    new_stateFeatures(1,2) = stateFeatures(4,realAgentLocation(2));
                    new_stateFeatures(1,3) = stateFeatures(4,realAgentLocation(2)+1);
                end
            
            %Do convolutional mutipplication to get three new features
%                 new_stateFeatures(1,1) = sum( sum(filter .* stateFeatures(:,1:3)));
%                 new_stateFeatures(1,2) = sum( sum(filter .* stateFeatures(:,2:4)));
%                 new_stateFeatures(1,3) = sum( sum(filter .* stateFeatures(:,3:5)));
            end
            
             % If you need to keep track of agent movement history:
            agentMovementHistory = zeros(episodeLength+1, 2) ;
            agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;

            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* new_stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values);

       
            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;

            next_stateFeatures = MDP.getStateFeatures(realAgentLocation);
            
            %pick three features based on the location of the car.
            if realAgentLocation(2) == 1
                    newNext_stateFeatures(1,1) = 0;
                    newNext_stateFeatures(1,2) = next_stateFeatures(4,realAgentLocation(2));
                    newNext_stateFeatures(1,3) = next_stateFeatures(4,realAgentLocation(2)+1);
                elseif realAgentLocation(2) == 5
                    newNext_stateFeatures(1,1) = next_stateFeatures(4,realAgentLocation(2)-1);
                    newNext_stateFeatures(1,2) = next_stateFeatures(4,realAgentLocation(2));
                    newNext_stateFeatures(1,3) = 0;
                else
                    newNext_stateFeatures(1,1) = next_stateFeatures(4,realAgentLocation(2)-1);
                    newNext_stateFeatures(1,2) = next_stateFeatures(4,realAgentLocation(2));
                    newNext_stateFeatures(1,3) = next_stateFeatures(4,realAgentLocation(2)+1);
            end

            
            for action = 1:3
                next_action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* newNext_stateFeatures ) );
            end % for each possible action
            [~, next_actionTaken] = max(next_action_values);


            prediction =  sum ( sum( weight(:,:,actionTaken) .* new_stateFeatures ) );
            target = agentRewardSignal + discountFactor_gamma * ...
                sum ( sum( weight(:,:,next_actionTaken) .* newNext_stateFeatures ) );
            
             %epsilon_greedy
            random = rand();
            candidate = [1,2,3];
            Locate=find(next_actionTaken); 
            candidate(Locate)=[];  
            
            %when the random value epsilon_greedy is less than the threshold,
            %do the random action
            if random < 2 * epsilon_greedy / 3 
                prob = rand();
                if prob >=0.5
                    pick = 1;
                else
                    pick = 2;
                end
                next_actionTaken = candidate(pick);
                %If one of those states would be outside the map, 
                %the next state will be the one above the current one
                if (realAgentLocation(2) == 1 && next_actionTaken == 1) || ...
                        (realAgentLocation(2) == 5 && next_actionTaken == 3)
                        next_actionTaken = 2;
                end
            end
            
            %Use changing learning rate
            learning_rate = 0.001/episode;

            delta_w = learning_rate * (target - prediction) * new_stateFeatures;
            

            %update weight
            weight(:,:,actionTaken) = weight(:,:,actionTaken) + delta_w;
            
            %update policy
            Q_test1 = weight
            
            new_stateFeatures = newNext_stateFeatures;

        end

        currentMap = MDP ;
        agentLocation = realAgentLocation ;
        
    end % for each episode
    
    for episode = 1:100
    
        %%
        currentTimeStep = 0 ;
        MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
            blockSize, noCarOnRowProbability, ...
            probabilityOfUniformlyRandomDirectionTaken, rewards );
        currentMap = MDP ;
        agentLocation = currentMap.Start ;
        startingLocation = agentLocation ; % Keeping record of initial location.
        realAgentLocation = agentLocation ; % The location on the full test map.
        Return = 0;
        for i = 1:episodeLength
            stateFeatures = MDP.getStateFeatures(realAgentLocation) % dimensions are 4rows x 5columns
        
            new_stateFeatures(1,1) = sum( sum(filter .* stateFeatures(:,1:3)));
            new_stateFeatures(1,2) = sum( sum(filter .* stateFeatures(:,2:4)));
            new_stateFeatures(1,3) = sum( sum(filter .* stateFeatures(:,3:5)));
            

            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* new_stateFeatures ) );
            end % for each possible action
            [~, actionTaken] = max(action_values)

            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;
            Return = Return + agentRewardSignal;
            [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
                MDP, realAgentLocation, blockSize );
            currentMap = viewableGridMap ; %#ok<NASGU>
            refreshScreen
            pause(0.15)
        end
        currentMap = MDP ;
        agentLocation = realAgentLocation ;
        printAgentTrajectory
        pause(1)

    end % for each episode
end


    

