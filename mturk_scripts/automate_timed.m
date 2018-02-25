input=fopen('input_timed.txt');
C = textscan(input, '%s', 'delimiter', '\n');
fclose(input);
content = char(C{1});
ouf=fopen('output_timed.txt','w');
for i=1:59
    for k=1:size(content,1)
        output=strrep(content(k,:),'choice',['choice_' num2str(i)]);
        output=strrep(output,'image_A_url',['image_A_url_' num2str(i)]);
        output=strrep(output,'image_B_url',['image_B_url_' num2str(i)]);
        output=strrep(output,'time',['time_' num2str(i)]);
        output=strrep(output,'dot_order',['dot_order_' num2str(i)]);
        output=strrep(output,'hiddenA',['hiddenA_' num2str(i)]);
        output=strrep(output,'hiddenB',['hiddenB_' num2str(i)]);
        fprintf(ouf,'%s\n',strtrim(output));
    end
    fprintf(ouf,'\n');
end
fclose(ouf);

rng('shuffle');
ouf=fopen('data_timed.csv','w');
tm=[300 400 600 1200 2400 3600 4800];

% mturk variables
for j=1:59
    % each pairwise comparison is associated with 4 variables:
    % imgA, imgB, time, and dot appearance order
    fprintf(ouf,['image_A_url_' num2str(j) ',' 'image_B_url_' num2str(j) ',' 'time_' num2str(j) ',' 'dot_order_' num2str(j)]);
    if(j==59)
        fprintf(ouf,'\n');
    else
        fprintf(ouf,',');
    end
end

% 50 HITs, 59 pairs of images in each
% out of that 59, 3 are sentinel pairs and 3 are warmup pairs
for i=1:50
	lst=randperm(59);
    % take textures 36, 37, and 19 and use them for warmup section
    lst(lst==36) = [];
    lst(lst==37) = [];
    lst(lst==19) = [];
    for c = [36 37 19]
        if c == 36
            t = tm(7);
            dot_ord = '12';
        elseif c == 37
            t = tm(4);
            dot_ord = '21';
        elseif c == 19
            t = tm(1);
            dot_ord = '12';
        end
        if t > 1200
            fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/1_%d/%06d.gif,', t, c);
            fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/4_%d/%06d.gif,', t, c);
        else
            fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/1_1200/%06d.gif,', c);
            fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/4_1200/%06d.gif,', c);
        end
        fprintf(ouf,'%d,', t);
        fprintf(ouf,'%s', dot_ord);
        fprintf(ouf,',');
    end
    inds=randperm(56);
    % 56 because we took 3 out from 59 for warmup
	for j=1:56
        % sample texture index
        c=lst(j);
        % randomize pair order
        if(rand()<0.5)
            order=[1 2];
        else
            order=[2 1];
        end
        % randomize dot appearance order
        if(rand()<0.5)
            dot_ord = '12';
        else
            dot_ord = '21';
        end
        % sentinel
        if(inds(j)>53)
            % take 3 warmup textures out and only pick a non-warmup one
            for k=order
                % obviously good example
                if(k==1)
                    fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/1_4800/%06d.gif,',c);
                end
                % obviously bad example
                if(k==2)
                    fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/3_4800/%06d.gif,',c);
                end
            end
            fprintf(ouf,'%d,', 4800); % time is 4.8 seconds for sentinel example
            fprintf(ouf,'%s', dot_ord); % dot order picked to be either 12 or 21 i.e., 1 then 2, or 2 then 1
        % not sentinel
        else
            t = tm(randi(7));
            for k=order
                if(k==1)
                    % output
                    if t > 1200
                        fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/4_%d/%06d.gif,', t, c);
                    else
                        fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/4_1200/%06d.gif,',c);
                    end
                end
                if(k==2)
                    % target
                    if t > 1200
                        fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/1_%d/%06d.gif,', t, c);
                    else
                        fprintf(ouf,'https://ryersonvisionlab.github.io/two-stream-projpage/textures/gifs/1_1200/%06d.gif,',c);
                    end
                end
            end
            fprintf(ouf,'%d,', t); % time picked between index 1 and 7 of tm
            fprintf(ouf,'%s', dot_ord); % dot order picked to be either 12 or 21 i.e., 1 then 2, or 2 then 1
        end
        if(j==56)
            fprintf(ouf,'\n'); % onto the next HIT (one HIT per row of vars)
        else
            fprintf(ouf,','); % onto the next set of vars (imgA, imgB, time)
        end
	end
end
fclose(ouf);
