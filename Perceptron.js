var NaturalIntelligence = (function () {
    "use strict";
    var _self = {};
    _self.Perceptron = function() {
        var me = {};
        me.layers = new _self.Layer();
        me.activate = function(inputs) {
            var inputLayer = me.layers[0];
            var outputLayer = me.layers[me.layers.length - 1];
            for(var i = 0; i < inputs.length; i++) {
                inputLayer[i].signal(inputs[i]);
            }
            var results = new Array(outputLayer.length);
            for(var o = 0; o < outputLayer.length; o++) {
                results[o] = outputLayer[o].activation;
            }
            return results;
        };

        me.create = function() {
            var below = new _self.Layer(me.layers.length < 1 ? null : me.layers[me.layers.length - 1]);
            me.layers.push(below);
            return below;
        };
        return me;
    };

    _self.ErrorBackPropagation = function (perceptron, stoppingConditionCallback, stoppingConditionInterval) {
        if (arguments.length == 2) {
            throw 'If you specify a stopping condition function you must also specify the number of training cycles that should be completed before it is called.'
        }
        var me = {};
        var _outputLayer = function() {
            return perceptron.layers[perceptron.layers.length - 1];
        };
        var _calculateHiddenError = function(hiddenUnit, outputErrors) {
            var hiddenUnitActivation = hiddenUnit.activation;
            var outputErrorTimesWeightCumulative = 0;
            var numberOfOutputUnits = outputErrors.length;

            for (var i = 0; i < numberOfOutputUnits; i++)
            {
                var weightStrength = hiddenUnit.outgoingWeights[i].strength;
                var error = outputErrors[i];
                outputErrorTimesWeightCumulative += error * weightStrength;
            }

            return hiddenUnitActivation * (1 - hiddenUnitActivation) * outputErrorTimesWeightCumulative;
        };
        var _resetActivations = function()
        {
            for(var i = 0; i < me.perceptron.layers.length; i++) {
                me.perceptron.layers[i].resetActivations();
            }
        };
        me.calculateCycleError = function(exemplar) {
            var sumErr = 0;
            var outputLayer = _outputLayer();
            var numberOfOutputUnits = outputLayer.length;
            for (var i = 0; i < numberOfOutputUnits; i++)
            {
                sumErr += Math.pow(exemplar.desired[i] - outputLayer[i].activation, 2);
            }
            return 0.5 * sumErr;
        };
        me.calculateHiddenLayerErrors = function(outputErrors, layer) {
            var hiddenErrors = [];
            for (var h = 0; h < layer.length; h++)
            {
                var hiddenUnit = layer[h];
                var hiddenError = _calculateHiddenError(hiddenUnit, outputErrors);
                hiddenErrors.push(hiddenError);
            };
            return hiddenErrors;
        };
        me.calculateOutputError = function(desiredOutput, actualOutput) {
            return (desiredOutput - actualOutput) * actualOutput * (1 - actualOutput);
        };
        me.calculateTopLayerErrors = function(exemplar) {
            var errors = [];
            var outputLayer = _outputLayer();
            var numberOfOutputUnits = outputLayer.length;
            for (var i = 0; i < numberOfOutputUnits; i++)
            {
                var activation = outputLayer[i].activation;
                var desired = exemplar.desired[i];
                errors.push(me.calculateOutputError(desired, activation));
            }
            return errors;
        };
        me.train = function(trainingData)
        {
            //return;
            if (typeof me.errorMax == 'undefined' || me.errorMax == null) {
                throw 'errorMax should be defined. Training aborted to prevent infinite loop.';
            }
            while (me.trainCycle(trainingData) == false) {
            }
        };
        var cycles = 0;
        me.trainCycle = function(trainingData)
        {
            var total = 0;
            for(var i = 0; i < trainingData.length; i++) {
                total += me.trainSingle(trainingData[i]);
            }
            cycles++;
            if (stoppingConditionInterval > 0 && cycles >= stoppingConditionInterval) {
                if (stoppingConditionCallback && stoppingConditionCallback()) {
                    throw 'The stopping condition was reached. Training has ended';
                }
                cycles = 0;
            }
            return total < me.errorMax;
        };

        me.trainSingle = function (exemplar) {
            me.perceptron.activate(exemplar.inputs);

            var outputErrors = me.calculateTopLayerErrors(exemplar);
            var hiddenToOutputWeightChanges = me.weightChanger.changeWeights(outputErrors, me.perceptron.layers[me.perceptron.layers.length - 2].getActivations(), _outputLayer());

            var lastHiddenIndex = me.perceptron.layers.length - 2;
            for (var i = lastHiddenIndex; i > 0; i--) {
                var upperLayer = me.perceptron.layers[i];
                var lowerLayer = me.perceptron.layers[i - 1];
                var upperErrors = me.calculateHiddenLayerErrors(outputErrors, upperLayer);
                var inputToHiddenWeightChanges = me.weightChanger.changeWeights(upperErrors, lowerLayer.getActivations(), upperLayer);
                outputErrors = upperErrors;
            }

            var sumErr = me.calculateCycleError(exemplar);

            _resetActivations();

            return sumErr;
        };

        me.perceptron = perceptron;
        me.errorMax = null;
        me.weightChanger = new _self.ErrorBackPropagationWeightChanger();
        me.weightChanger.learningConstant = 0.1;
        return me;
    };
    _self.ErrorBackPropagationWeightChanger = function(settings) {
        var me = {};
        var _uncommittedChanges = [];
        var _previousCycleChanges = [];
        me.learningConstant = 1;
        me.momentumConstant = 0;
        me.useBias = false;
        me.numberOfLayers = 0;
        if(settings) {
            me.numberOfLayers += settings.hiddenLayers || 0;
            me.numberOfLayers += settings.inputLayers || 0;
            me.numberOfLayers += settings.outputLayers || 0;

            if(typeof settings.learningConstant != 'undefined') {
                me.learningConstant = settings.learningConstant;
            }
            if(typeof settings.useBias != 'undefined') {
                me.useBias = settings.useBias;
            }
            if(typeof settings.momentumConstant != 'undefined') {
                me.momentumConstant = settings.momentumConstant;
            }
        }
        me.enqueue = function(change) {
            _uncommittedChanges.push(change);

            //If this is the last change of the cycle then push to previous changes queue
            if (_uncommittedChanges.length + 1 === me.numberOfLayers)
            {
                while (_uncommittedChanges.length > 0)
                {
                    _previousCycleChanges.push(_uncommittedChanges.pop());
                }
            }
        };
        me.dequeue = function() {
            return _previousCycleChanges.length === 0 ? null : _previousCycleChanges.pop();
        };
        me.calculateInputToOutputWeightChanges = function(outputErrors, inputActivations) {
            var setsOfChanges = [];
            for (var o = 0; o < outputErrors.length; o++)
            {
                var weightChanges = [];
                for (var h = 0; h < inputActivations.length; h++)
                {
                    var weightChange = me.calculateInputToOutputWeightChange(inputActivations[h], outputErrors[o]);
                    weightChanges.push(weightChange);
                }
                setsOfChanges.push(weightChanges);
            }
            return setsOfChanges;
        };
        me.calculateInputToOutputWeightChange = function(activation, error) {
            return me.learningConstant * error * activation;
        };
        me.changeWeightStrengths = function(weightChanges, upperLayer) {
            var lastWeightChanges = me.dequeue();
            for (var iUpper = 0; iUpper < weightChanges.length; iUpper++)
            {
                var weightChangesForOutput = weightChanges[iUpper];
                for (var w = 0; w < weightChangesForOutput.length; w++)
                {
                    var weightChange = weightChangesForOutput[w];
                    var momentumChange = 0;
                    if (me.momentumConstant > 0 && lastWeightChanges != null)
                    {
                        momentumChange = me.momentumConstant * lastWeightChanges[iUpper][w];
                    }
                    var weightToChange = upperLayer[iUpper].incomingWeights[w];
                    weightToChange.strength += weightChange + momentumChange;
                }
            }
        };
        me.changeWeights = function(errors, inputActivations, upperLayer)
        {
            var inputToHiddenWeightChanges = me.calculateInputToOutputWeightChanges(errors, inputActivations);
            if (me.useBias)
            {
                for (var i = 0; i < upperLayer.length; i++)
                {
                    upperLayer[i].addInput(errors[i]);
                }
            }
            me.changeWeightStrengths(inputToHiddenWeightChanges, upperLayer);
            return inputToHiddenWeightChanges;
        };
        return me;
    };
    _self.Exemplar = function(inputs, desired) {
        return {
            inputs : inputs || [],
            desired : desired || []
        };
    };
    _self.PerceptronFactory = {
        create : function(settings) {
            var addLayer = function(numberOfUnits, activationFunction, createInputArea, perceptron) {
                var layer = perceptron.create();
                for (var i = 0; i < numberOfUnits; i++)
                {
                    var unit = new _self.StandardUnit();
                    unit.activationFunction = activationFunction;
                    unit.inputArea = (createInputArea || _self.SummedInputArea)();
                    layer.add([unit]);
                }
            };

            settings.hiddenLayers = settings.hiddenLayers || [];
            settings.inputActivationFunction = settings.inputActivationFunction || new _self.DummyActivationFunction();
            settings.hiddenActivationFunction = settings.hiddenActivationFunction || new _self.UnipolarActivationFunction();
            settings.outputActivationFunction = settings.outputActivationFunction || (settings.hiddenLayers.length > 0 ? new _self.UnipolarActivationFunction() : new _self.ThresholdActivationFunction(1));

            var perceptron = new _self.Perceptron();
            addLayer(settings.inputUnits, settings.inputActivationFunction, settings.inputUnitsInputAreaCreator, perceptron);

            //Add n hidden layers
            for (var h = 0; h < settings.hiddenLayers.length; h++)
            {
                var numberOfUnits = settings.hiddenLayers[h];
                //Number of units in each hidden layer
                addLayer(numberOfUnits, settings.hiddenActivationFunction, settings.hiddenUnitsInputAreaCreator, perceptron);
            }

            addLayer(settings.outputUnits, settings.outputActivationFunction, settings.outputUnitsInputAreaCreator, perceptron);
            return perceptron;
        }
    };
    _self.PerceptronRule = function (perceptron, stoppingConditionCallback, stoppingConditionInterval) {
        if (arguments.length == 2) {
            throw 'If you specify a stopping condition function you must also specify the number of training cycles that should be completed before it is called.'
        }
        var me = {};
        me.perceptron = perceptron;
        var _outputLayer = function() { return me.perceptron.layers[me.perceptron.layers.length - 1]; };
        var _inputLayer = function() { return me.perceptron.layers[0]; };
        me.weightChanger = null;
        me.train = function (trainingData) {
            while (me.trainCycle(trainingData) == false) { }
        };
        var cycles = 0;
        me.trainCycle = function(trainingData) {
            var results = true;
            for (var t = 0; t < trainingData.length; t++) {
                var td = trainingData[t];
                results &= me.trainSingle(td);
            }
            cycles++;
            if (stoppingConditionInterval > 0 && cycles >= stoppingConditionInterval) {
                if (stoppingConditionCallback && stoppingConditionCallback()) {
                    throw 'The stopping condition was reached. Training has ended';
                }
                cycles = 0;
            }
            return results;
        };
        me.trainSingle = function(exemplar) {
            me.perceptron.activate(exemplar.inputs);

            var outputsAreCorrect = me.calculateIfUnitsRespondCorrectly(exemplar);

            var allCorrect = true;
            for (var c = 0; c < outputsAreCorrect.length; c++) {
                var isCorrect = outputsAreCorrect[c];
                allCorrect &= isCorrect;
            }
            if(allCorrect == false)
            {
                //Apply weight changes
                var outputSignals = [];
                var outputLayer = _outputLayer();
                for(var o = 0; o < outputLayer.length; o++)
                {
                    var err = outputLayer[o].activation - exemplar.desired[o];
                    outputSignals.push(err);
                }
                me.weightChanger.changeWeights(outputSignals, exemplar.inputs, outputLayer);
            }

            me.resetActivations();

            return allCorrect;
        };
        me.resetActivations = function () {
            for (var i = 0; i < me.perceptron.layers.length; i++)
            {
                me.perceptron.layers[i].resetActivations();
            }
        };
        me.calculateIfUnitsRespondCorrectly = function(exemplar) {
            var results = [];
            var outputLayer = _outputLayer();
            for (var i = 0; i < outputLayer.length; i++) {
                var activation = outputLayer[i].activation;
                var desired = exemplar.desired[i];
                results.push(desired == activation);
            }
            return results;
        }
        return me;
    };
    _self.PerceptronRuleWeightChanger = function() {
        var me = {};
        me.changeWeights = function(outputErrors, inputActivations, upperLayer) {
            var inputToHiddenWeightChanges = me.calculateInputToOutputWeightChanges(outputErrors, inputActivations);
            me.changeWeightStrengths(inputToHiddenWeightChanges, upperLayer);
            return inputToHiddenWeightChanges;
        };
        me.changeWeightStrengths = function(weightChanges, upperLayer) {
            for (var iUpper = 0; iUpper < weightChanges.length; iUpper++) {
                var weightChangesForOutput = weightChanges[iUpper];
                for (var w = 0; w < weightChangesForOutput.length; w++) {
                    var weightChange = weightChangesForOutput[w];
                    var weightToChange = upperLayer[iUpper].incomingWeights[w];
                    weightToChange.strength += weightChange;
                }
            }
        };
        me.calculateInputToOutputWeightChanges = function(outputErrors, inputActivations) {
            var setsOfChanges = [];
            for (var o = 0; o < outputErrors.length; o++) {
                var weightChanges = [];
                for (var h = 0; h < inputActivations.length; h++) {
                    var weightChange =  me.calculateInputToOutputWeightChange(inputActivations[h], outputErrors[o]);
                    weightChanges.push(weightChange);
                }
                setsOfChanges.push(weightChanges);
            }
            return setsOfChanges;
        };
        me.calculateInputToOutputWeightChange = function(inputActivation, outputError) {
            var multiplier = 0;
            if (outputError < 0) { multiplier = 1; }
            else if (outputError > 0) { multiplier = -1; }
            return multiplier * me.learningConstant * inputActivation;
        };
        return me;
    };

    var _initializeWeightsRandomly = function(perceptron, weightStrengthFunction) {
        for (var i = 0; i < perceptron.layers.length - 1; i++) {
            var layer = perceptron.layers[i];
            for (var j = 0; j < layer.length; j++) {
                for (var w = 0; w < layer[j].outgoingWeights.length; w++) {
                    layer[j].outgoingWeights[w].strength = weightStrengthFunction();
                }
            }
        }
    };

    _self.PerceptronTrainer = function() {
        var me = {};
        me.learningRule = null;
        me.perceptron = null;
        me.weightStrengthCreator = null;
        me.train = function(trainingData) {
            _initializeWeightsRandomly(me.perceptron, me.weightStrengthCreator);
            me.learningRule.train(trainingData);
        };
        return me;
    };
    _self.PerceptronTrainingFactory = {
        create : function(settings)
        {
            var perceptron = _self.PerceptronFactory.create(settings);
            var trainer = new _self.PerceptronTrainer();
            trainer.perceptron = perceptron;
            trainer.learningRule = new _self.ErrorBackPropagation(perceptron);
            trainer.weightStrengthCreator = settings.weightStrengthCreator;
            return trainer;
        }
    };
    //+ Jonas Raoni Soares Silva
    //@ http://jsfromhell.com/array/shuffle [v1.0]
    function shuffle(o){ //v1.0
        for(var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
        return o;
    };
    _self.RandomExemplarEnumerator = function(exemplars) {
        return shuffle(examplars);
    };
    _self.ThresholdActivationFunction = function(threshold) {
        var me = {};
        me.threshold = threshold || 0;
        me.activate = function(input) {
            return input >= me.threshold ? 1 : 0;
        };
        return me;
    };
    _self.UnipolarActivationFunction = function(steepness) {
        var me = {};
        me.steepness = typeof steepness != 'undefined' ? steepness : 1;
        me.activate = function(input) {
            return 1 / (Math.pow(Math.exp(1), -me.steepness * input) + 1);
        };
        return me;
    };
    _self.DummyActivationFunction = function() {
        return {
            activate : function(input) { return input; }
        };
    };
    _self.Layer = function(lowerLayer, upperLayer) {
        var me = [];
        me.layerBelow = lowerLayer || null;
        if(lowerLayer) {
            lowerLayer.layerAbove = me;
        }
        me.layerAbove = upperLayer || null;
        if(upperLayer) {
            upperLayer.layerBelow = me;
        }
        me.resetActivations = function() {
            for(var u = 0; u < me.length; u++) {
                me[u].reset();
            }
        };
        me.addDisconnected = function(units) {
            for (var u = 0; u < units.length; u++) {
                me.push(units[u]);
            }
        };
        me.add = function(units) {
            me.addDisconnected(units);
            for(var i = 0; i < units.length; i++) {
                units[i].connectTo(me.layerBelow, me.layerAbove);
            }
            return me;
        };
        me.forEach = function(callback) {
            for(var u = 0; u < me.length; u++) {
                callback(me[u]);
            }
        };
        me.getActivations = function() {
            var results = [];
            for(var u = 0; u < me.length; u++) {
                results.push(me[u].activation);
            }
            return results;
        };
        return me;
    };
    _self.MultipliedInputArea = function () {
        var me = {};
        me.total = 0;
        me.reset = function() {
            me.total = 0;
        };
        me.addInput = function(value) {
            me.total += value;
        };
        return me;
    };
    _self.RandomWeightStrengthFactory = function() {
        var me = {};
        me.startRange = -2;
        me.endRange = 2;
        me.getStrength = function() {
            var difference = Math.abs(me.endRange - me.startRange);
            return Math.random() * difference + me.startRange;
        };
        return me;
    };
    _self.StandardUnit = function() {
        var me = {};
        var signalsReceived = 0;
        var subscribers = [];
        me.incomingWeights = [];
        me.outgoingWeights = [];
        me.activation = null;
        me.activationFunction = null;
        me.inputArea = null;
        me.addInput = function(input) {
            me.inputArea.addInput(input);
        };
        me.reset = function() {
            me.activation = null;
        };
        me.signal = function(value) {
            signalsReceived++;
            me.addInput(value);
            if(signalsReceived >= me.incomingWeights.length) {
                me.fire();
                signalsReceived = 0;
                me.inputArea.reset();
            }
        };
        me.subscribe = function(subscriber) {
            subscribers.push(subscriber);
        };
        me.fire = function() {
            me.activation = me.activationFunction.activate(me.inputArea.total);
            for(var s = 0; s < subscribers.length; s++) {
                subscribers[s].signal(me.activation);
            }
        };
        me.connectTo = function(unitsBelow, unitsAbove) {
            if (unitsBelow != null) {
                for(var ub = 0; ub < unitsBelow.length; ub++) {
                    var connection = new _self.StandardWeight(unitsBelow[ub], me);
                    unitsBelow[ub].subscribe(connection);
                }
            }

            if (unitsAbove != null) {
                for(var ua = 0; ua < unitsAbove.length; ua++) {
                    var connection = new _self.StandardWeight(me, unitsAbove[ua]);
                    me.subscribe(connection);
                }
            }
        };
        return me;
    };
    _self.StandardWeight = function(incomingUnit, outgoingUnit) {
        var me = {};
        me.incomingUnit = incomingUnit;
        me.incomingUnit.outgoingWeights.push(me);
        me.outgoingUnit = outgoingUnit;
        me.outgoingUnit.incomingWeights.push(me);
        me.strength = 0;
        me.signal = function(value) {
            me.outgoingUnit.signal(value * me.strength);
        };
        return me;
    };
    _self.SummedInputArea = function() {
        var me = {};
        me.total = 0;
        me.reset = function() {
            me.total = 0;
        };
        me.addInput = function(value) {
            me.total += value;
        };
        return me;
    };
    return _self;
})();