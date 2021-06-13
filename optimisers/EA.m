% bestxSofar: the best solution found by your algorithm
% recordedAvgY: array of  average fitnesses of each generation
% recordedBestY: array of best fitnesses of each generation
function [bestxSofar, recordedAvgY, recordedBestY]=EA(funcName,n,lb,ub,nbEvaluation)
  warning on MATLAB:divideByZero
    if nargin < 5
      error('input_example :  not enough input')
    end
  
    eval(sprintf('objective=@%s;',funcName)); % Do not delete this line
    % objective() is the evaluation function, for example : fitness = objective(x) 
  
    %%%%%%%%%%%% IFEP with crossover on all benchmark,simple crossover
    %% Initialise variables and parameters
    % Super parameters of EA
    Mu = 50;
    Tau=1/sqrt(2*sqrt(n));
    Tau2=1/sqrt(2*n);
    LocalsearchThreshold = 1/(ub-lb)/Mu/5;
    % Super parameters of Selection
    Tournamentsize = 10; 
    % Super parameters of Crossover
    Alpha = 0.5;
    % Super parameters of Mutation
    Etainitial = 3.0;
    % Variables of EA
    nbGen = 1;
    nbEval = 0;
    % Variables of Mutation
    etaMatrix = Etainitial*ones(Mu,n);
    etaMatrix = boundData(etaMatrix,10^(-3));
  
    %% Initialise a population 
    population = lb + (ub-lb)*rand(Mu,n);
    population = boundData(population,lb,ub);
    
    %% Evaluate the initial population
    fitness = zeros(Mu,1);
    for i = 1:Mu
        fitness(i) = -objective(population(i,:)); 
        nbEval = nbEval + 1;
    end
    
    %% Record some data
    recordedAvgY = zeros(1,nbEvaluation);
    recordedBestY = zeros(1,nbEvaluation);
    [bestFit,bestFitId] = max(fitness);
    bestxSofar = population(bestFitId,:);
    recordedBestY(nbGen) = bestFit;
    recordedAvgY(nbGen) = mean(fitness); 
  
    parentsId = randperm(Mu);
    parents1Id = parentsId(1:Mu/2);
    parents2Id = parentsId((Mu/2+1):end);
    parents1 = population(parents1Id,:);
    parents2 = population(parents2Id,:);
    
    offspringsCauchy = zeros(Mu,n);
    offspringsCauchy1 = offspringsCauchy(1:Mu/2,:);
    offspringsCauchy2 = offspringsCauchy(1:Mu/2,:);
    fitOfOffspringsCauchy = zeros(Mu,1);
    etaMatrixOffspringsCauchy = zeros(Mu,n);
    
    offspringsGaussian = zeros(Mu,n);
    offspringsGaussian1 = offspringsCauchy(1:Mu/2,:);
    offspringsGaussian2 = offspringsCauchy(1:Mu/2,:);
    fitOfOffspringsGaussian = zeros(Mu,1);
    etaMatrixOffspringsGaussian = zeros(Mu,n);
    
    % Start the loop
    while (nbEval<nbEvaluation)    
      
      %% Crossover (Simple Arithmetic Recombination)
      % All individuals are used as parents, so it's not necessary for Parents-Selection is 
      % Cauchy
      if nbGen>1
        if (abs(recordedBestY(nbGen)-recordedBestY(nbGen-1)) > LocalsearchThreshold) && (recordedBestY(nbGen)>recordedBestY(nbGen-1))
          disp('no recombination');
          i = [1:(Mu/2)]';
          singleID = randi(n);
          offspringsCauchy1(i,1:(singleID)) = parents1(i,1:(singleID));
          offspringsCauchy1(i,singleID+1:end) = Alpha * parents1(i,singleID+1:end) + (1-Alpha) * parents2(i,singleID+1:end);
          offspringsCauchy2(i,1:(singleID)) = parents2(i,1:(singleID));
          offspringsCauchy2(i,singleID+1:end) = Alpha * parents2(i,singleID+1:end) + (1-Alpha) * parents1(i,singleID+1:end); 
          offspringsCauchy = [offspringsCauchy1;offspringsCauchy2];
          % Gaussian
          i = [1:(Mu/2)]';
          singleID = randi(n);
          offspringsGaussian(i,1:(singleID)) = parents1(i,1:(singleID));
          offspringsGaussian1(i,singleID+1:end) = Alpha * parents1(i,singleID+1:end) + (1-Alpha) * parents2(i,singleID+1:end);
          offspringsGaussian2(i,1:(singleID)) = parents2(i,1:(singleID));
          offspringsGaussian2(i,singleID+1:end) = Alpha * parents2(i,singleID+1:end) + (1-Alpha) * parents1(i,singleID+1:end);
          offspringsGaussian = [offspringsGaussian1;offspringsGaussian2];
        end
      end

      %% Mutation (Nonuniform Mutation with Cauchy Distribution)
      if nbGen>1
        if (abs(recordedBestY(nbGen)-recordedBestY(nbGen-1))<LocalsearchThreshold) && (recordedBestY(nbGen)>recordedBestY(nbGen-1))
          disp('add local search')
          etaMatrix = etaMatrix./nbGen;
        end
      end
      % Cauchy
      offspringsCauchy = population + etaMatrix .* cauchy(Mu,n);
      offspringsCauchy = boundData(offspringsCauchy,lb,ub);
      RandNorDistr = randn(Mu,1).*ones(1,n);
      etaMatrixOffspringsCauchy = etaMatrix.*exp(Tau2 * RandNorDistr + Tau * randn(Mu,n));
      etaMatrixOffspringsCauchy = boundData(etaMatrixOffspringsCauchy,10^(-3));
      for i = 1:Mu
        fitOfOffspringsCauchy(i) = -objective(offspringsCauchy(i,:));
        nbEval = nbEval + 1;
      end
      
      % Gaussian
      offspringsGaussian = population + etaMatrix .* randn(Mu,n);
      offspringsGaussian = boundData(offspringsGaussian,lb,ub);
      RandNorDistr = randn(Mu,1).*ones(1,n);
      etaMatrixOffspringsGaussian = etaMatrix.*exp(Tau2 * RandNorDistr + Tau * randn(Mu,n));
      etaMatrixOffspringsGaussian = boundData(etaMatrixOffspringsGaussian,10^(-3));
      for i = 1:Mu
        fitOfOffspringsGaussian(i) = -objective(offspringsGaussian(i,:)); 
        nbEval = nbEval + 1;
      end

      %% Local search
      [sortedFit,sortedFitId] = sort(fitness,'descend');
      bestIndividual = population(sortedFitId(1:20),:);
      % bestIndividual = bestIndividual + randn(20,n);
      bestIndividual = bestIndividual + normrnd(0,1,[20 n]);
      bestIndividual = boundData(bestIndividual,lb,ub);
      bestIndividualFit = zeros(20,1);
      for i = 1:20
        bestIndividualFit(i) = -objective(bestIndividual(i,:));
        nbEval = nbEval + 1;
      end

      %% Survivor Selection (Tournament Selection) 
      allIndividuals = [population;offspringsCauchy;offspringsGaussian;bestIndividual];
      fitAllIndividuals = [fitness;fitOfOffspringsCauchy;fitOfOffspringsGaussian;bestIndividualFit];
      etaMatrixAllIndividuals = [etaMatrix;etaMatrixOffspringsCauchy;etaMatrixOffspringsGaussian;randn(20,n)];
      nbWin = zeros(3*Mu+20,1);
      randId = randperm(3*Mu+20);
      opponentsID = randId(1:Tournamentsize);
      for i = 1:(3*Mu+20)
        listWin = find(fitAllIndividuals(opponentsID) < fitAllIndividuals(i));
        nbWin(i) = length(listWin);
      end
      
      [sortedNbWin,sortedNbWinId] = sort(nbWin,'descend');
      winnersId = sortedNbWinId(1:Mu);
      
      %% Update
      % Update population
      population = allIndividuals(winnersId,:);

      parentsId = randperm(Mu);
      parents1Id = parentsId(1:Mu/2);
      parents2Id = parentsId((Mu/2+1):end);
      parents1 = population(parents1Id,:);
      parents2 = population(parents2Id,:);

      fitness = fitAllIndividuals(winnersId);
      etaMatrix = etaMatrixAllIndividuals(winnersId,:);
      
      %% Recorde some data
      [bestFit,bestFitId] = max(fitness);
      bestxSofar = population(bestFitId,:);
      nbGen = nbGen + 1;
      recordedBestY(nbGen) = bestFit;
      recordedAvgY(nbGen) = mean(fitness); 
      fprintf(1,'%s\t %d/%d \t Gene:%d \t Avg:%d \t best:%d\n',funcName,nbEval,nbEvaluation,nbGen,recordedAvgY(nbGen),recordedBestY(nbGen));
    end 
    recordedBestY = recordedBestY(1:nbGen);
    recordedAvgY = recordedAvgY(1:nbGen);
    [bestFitVal,bestFitId] = max(fitness);
    bestx = population(bestFitId,:);
    
  end
  
  
  % function value = mycauchy(N,M,t)
  %   if nargin == 1
  %     value=tan((rand-1/2)*pi)*t;
  %   end
  %   if nargin == 2
  %     value=tan((rand(N)-1/2)*pi)*t;
  %   end
  %   if nargin == 3
  %     value=tan((rand(N,M)-1/2)*pi)*t;
  %   end
  % end
