%function to use the trained NN to classify the free-hand draw digit of 28
%by 28 pixel

function NN_try(NN)
% P = load('Training.mat');
%first create a matrix of 28x28 dimensional and completely black
num_pixel = 28;
digit = zeros(num_pixel,num_pixel);
f = figure(1); clf; hold on;
handle = imagesc(digit, [0,1]);
set(gca,'Xlim', [1,28]);
set(gca, 'Ylim', [1,28]);
ax = gca;
str_handle = title('Current Digit classified is ?');
colormap gray;
axis off;

%define an incremental value when drawn
increment = .05;
sigmoid = @(x) 1./(1+exp(-x));

%create a push button to allow user to draw
draw_button = uicontrol('Style', 'togglebutton', 'String', 'Draw',...
    'Position', [20 20 50 20], 'Callback', @draw);
%create a clear button to clear the canvas
clear_button = uicontrol('Style', 'pushbutton', 'String', 'Clear',...
    'Position', [400 20 120 20], 'Callback', @clear);
%create the a button to use neural network to classify the digit
nn_classify = uicontrol('Style', 'pushbutton', 'String', 'Classify NOW!',...
    'Position', [200 20 120 20], 'Callback', @classify);

    function draw(source, event)
        if(get(source, 'Value') == 1)
            set(source, 'FontWeight', 'bold');
            set(source, 'BackgroundColor', 'red');
            set(source, 'HitTest', 'off');
            %the following code is to allow user to freely draw on the figure
%             set(f,'windowbuttonmotionfcn',@wbmfcn);
            set(f,'windowbuttondownfcn',@wbdfcn);
            %Wait for right-click or double-click
            while ~strcmp(get(f,'SelectionType'),'alt') && ~strcmp(get(f,'SelectionType'),'open')
                drawnow;
            end
        else
            set(source, 'FontWeight', 'normal');
            set(source, 'BackgroundColor', 'default');
            set(f,'windowbuttonmotionfcn','');
            set(f,'windowbuttondownfcn','');
        end
    end
    function wbmfcn(varargin)
        if strcmp(get(gcf,'selectiontype'),'normal');
            a=get(ax,'currentpoint');
            a = round(a);
            %convert the (x,y) coordinate to the grid coordinate
            if(a(1,1) > 0 && a(1,1) <= num_pixel && a(1,2) > 0 && a(1,2) <= num_pixel)
                digit(a(1,1),a(1,2)) = digit(a(1,1),a(1,2)) + increment;
                %             display([round(a(1,1)),round(a(1,2))]);
                if(digit(a(1,1),a(1,2)) > 1)
                    digit(a(1,1),a(1,2)) = 1;
                end
                set(handle, 'CData', (digit'));
                drawnow;
            end
        end
    end
    function wbdfcn(varargin)
        if(strcmp(get(f,'SelectionType'),'alt') || ~strcmp(get(f,'SelectionType'),'open'))
            set(f,'windowbuttonmotionfcn','');
        end
        if(strcmp(get(f,'SelectionType'),'normal'))
            set(f,'windowbuttonmotionfcn',@wbmfcn);
        end
    end

    function clear(source, event)
        digit = digit.*0;
        set(handle, 'CData', digit);
        set(f,'windowbuttonmotionfcn','');
    end

    function classify(source,event)
        temp = flipud(get(handle, 'CData'));
        temp = reshape(temp',[num_pixel*num_pixel,1]);
        for i = 2:length(NN)
            temp = NN(i).weights*temp + NN(i).bias;
            temp = sigmoid(temp);
        end
        [~, ind_nn] = max(temp);
        set(str_handle, 'String', ['Current Digit classified is ', num2str(ind_nn-1)]);
    end
end