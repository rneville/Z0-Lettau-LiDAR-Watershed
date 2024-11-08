%%%%%%%%%%%%%%%%%%%%%%%%%%%%%0.1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7-NOV-2024: by Rachel Neville
% Updated from code by Ron Pasquini 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intro:  This program calculates aerodynamic roughness (z0) via Lettau's
%  Method.  The surface is separated into individual features using Fernand
%  Meyer's watershed algortithm.  Input data is surface height in a 
%  spreadsheet (xlsx file). User specifies filename, data location in the 
%  spreadsheet, data scale, filter option, and wind direction 
%
%  Overview: Lettau Aerodynamic Roughness (z0) calculated on LiDAR surface
% data
%
%  Input: file 'LiDAR_z0_runInfo01.txt' where user will specify xls data
%    file name, sheet, data location within spreadsheet, resolution (in
%    meters), if a lowpass filter should be applied, and wind direction.
%       
%  Method:
%   - Data is read in, optionally filtered with a low-pass filter, and mean
%    centered
%   - Watershed regions are computed
%   - Regions perpendicular to the wind are identified 
%   - Vetical silhouette areas of each watershed region are calculated by
%   using the subregion perpendicular to the given wind direction
%   - Feature height is computed as the height of the subregion 
%    perpendicularto the wind 
%   - WS Area computed from data in each watershed region (does not include
%    boundary pixels)
%   - Lettau z0 is calculated
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%============================================================= housekeeping

clear % clear memory
clc     % clear console
close ALL % close all figure windows

%--------------------------------------------------set some plot parameters

extremaParam = NaN;
fontTitle = 14;
fontAxes = 12;
% AzimElev = [135 45]; % a nice perspective viewing angle
AzimElev = [-30 45]; % another viewing angle
% custom colormap
cmap=[   18,39,64;
    27,72,94;
    50,107,119;
    86, 139, 135;
    128,174,154;
    181, 209, 174];
[Xc,Yc] = meshgrid([1:3],[1:50]);  % mesh of indices

cmap = interp2(Xc([1,10,20,30,40,50],:),Yc([1,10,20,30,40,50],:),cmap,Xc,Yc); % interpolate colormap
cmap=1/256*cmap;

wind_cmap=[155,192,164; 212, 114, 100]; % teal vs red for wind
wind_cmap=1/256*wind_cmap;

%--------------------------------------------------------------- greetings!
fprintf('\n');
fprintf('\n***************************************************************' );
fprintf('\n*    LiDAR Surface Aerodynamic Roughness (z0) Calculator      *' );
fprintf('\n*         using Lettau z0 method with watersheds              *' );
fprintf('\n***************************************************************' );
fprintf('\n');
%==========================================================================
%                                                   load LiDAR surface data
% data has been processed and saved as an Excel file
% current code uses an input file 'LiDAR_z0_runInfo01.txt', but can easily
% be switched to prompt for user input (commented out below)
%==========================================================================

infile = fopen('LiDAR_z0_runInfo01.txt','r'); % open file with read permission
cellarray = textscan(infile,'%q', 'CommentStyle','//'); % the %q picks off text between double quotes
%celldisp(cellarray)

dataFileName   = cellarray{1}{1};
dataTitleText = cellarray{1}{2};

fprintf('\nLoading LiDAR surface data:  ')
fprintf('\n    %s', dataTitleText)

sheet = str2num(cellarray{1}{3});
xlRange = cellarray{1}{4};

xyScale = str2num(cellarray{1}{5});

filterChoice = str2num(cellarray{1}{6});

dataSurface = xlsread(dataFileName, sheet, xlRange);

Wind_dir=str2num(cellarray{1}{7});

if size(Wind_dir)==[3 1]
elseif size(Wind_dir)==[1 3]
    Wind_dir = Wind_dir'
else
fprintf('Error. Wind direction must be a 3x1 vector.' )
end

fclose(infile);

%-------------------------------------------------------------------------------
% % OPTIONAL: Reworked to prompt user for inputs instead of reading in a 
% % txt file with this info (comment this out and uncomment following code 
% % block to read in txt file containing info)
%-------------------------------------------------------------------------------
% dataFileName=input('Enter xlsx data file name (with suffix): ' ,"s");
% dataTitleText1=input('Data Title for Plotting: ' ,"s");
%
% sheet=input('Excel sheet number where data is located: ');
% xlRange=input('Data spans cells as: upper_left:lower_right cell IDs: ', "s");
% xyScale=input('Data resolution as N meters/grid_step (decimal OK): ');
% filterChoice=input('Low pass filter the data? yes = 1, no = 0: ');
% Wind_dir=input('Wind direction 
% dataSurface = xlsread(dataFileName, sheet, xlRange); %read in data from the excel sheet
%
%==========================================================================
%                                                          prep the surface
%==========================================================================

numRows = length(dataSurface(1,:));
numCols = length(dataSurface(:, 1));
%--------------------------------------------------------------------------
%                                           FILTER whole surface if desired
%  see https://www.mathworks.com/help/images/ref/fspecial.html for more 
% filters
%--------------------------------------------------------------------------

if filterChoice == 1
    lpFilter = [1/9, 1/9, 1/9; 1/9, 1/9, 1/9; 1/9, 1/9, 1/9];  % low-pass filter
    dataFiltered = filter2(lpFilter, dataSurface);
    surfFiltered = dataFiltered(2:numCols - 2, 2:numRows - 2); % remove edges after filtering
else  % default to UNfiltered:
    surfFiltered = dataSurface;
end

%----------------------------------------------------- mean center the data
surfPreppedCentered = surfFiltered - mean(mean(surfFiltered)); % subtract mean

fullSurfDims = size(surfPreppedCentered);
fullX = fullSurfDims(2);
fullY = fullSurfDims(1);

%----------------------------- some preparations for Area of Interest (AOI)
aoiSurfReady = surfPreppedCentered;
aoiSurfDims = size(aoiSurfReady);
aoiX = aoiSurfDims(2);
aoiY = aoiSurfDims(1);

%--------------------------------------------------------------------------
%                 calculate & report extema stats for mean-centered surface
% (can easily be commented out)
%--------------------------------------------------------------------------

x = (1:aoiX);  % to work in grid-space
y = (1:aoiY);

[xSurf,ySurf] = meshgrid(x,y);  % create a grid of surface size

[zmax,imax,zmin,imin] = extrema2(aoiSurfReady);

numMaxs = length(zmax);
numMins = length(zmin);

fprintf('\n');
fprintf('\n***************************************************************')
fprintf('\n       Original Mean-Centered LiDAR Surface Data Summary       ')
fprintf('\n  (Dataset : %s)', dataTitleText)
fprintf('\n***************************************************************')
fprintf('\n  surface grid dims : %i x %i (x,y)', fullX, fullY)
fprintf('\n      xy-grid scale : %i (m/grid step)', xyScale)
fprintf('\n   x-dim (physical) : %i (m)', fullX*xyScale)
fprintf('\n   y-dim (physical) : %i (m)', fullY*xyScale)
fprintf('\n  # of local maxima : %i', numMaxs )
fprintf('\n  # of local minima : %i', numMins )
fprintf('\n      surface z-max : %5.3f (m)', max(zmax) )
fprintf('\n      surface z-min : %5.3f (m)', min(zmin) )
fprintf('\n       total relief : %5.3f (m) <---- RELIEF', max(zmax) - min(zmin) )
fprintf('\n     mean of maxima : %5.3f (m)' , mean(zmax) )
fprintf('\n     mean of minima : %5.3f (m)', mean(zmin) )
fprintf('\n        mean relief : %5.3f (m)', mean(zmax) - mean(zmin) );
fprintf('\n');

%==========================================================================
%                                               Begin Watershed computation
%==========================================================================
% Using the Fernand Meyer algorithm for Watershed computation
%--------------------------------------------------------------------------
WS=watershed(-aoiSurfReady); % since the watershed algortithm builds 
% watershed  regions associated with flow to basins (minima), we flip it 
% upside down to build regions around peaks

% imagesc(WS)

num_WS=max(max(WS)); % number of watershed regions

% initialize vestors to hold
WS_Sil_Area=zeros(num_WS,1);
[X,Y]=meshgrid(1:aoiX,1:aoiY);
X=reshape(X,[],1);
Y=reshape(Y,[],1);

% Compute the surface normal to every point on the surface
[Nx,Ny,Nz]=surfnorm(aoiSurfReady);
Wind_dir=Wind_dir/norm(Wind_dir); % normalizes wind direction vector  

% Project each surface normal onto the wind vector for the entire surface.
MN=([reshape(Nx,[],1),reshape(Ny,[],1), reshape(Nz,[],1)]*Wind_dir)/(Wind_dir'*Wind_dir);
MN=reshape(MN,aoiSurfDims); % MN is the signed size of the component of the surface normal vector that is aganist the wind

% Compute height and area of vertical silhouette of the subregion perpendicular
%  to the wind for each each watershed
for i=1:num_WS
    WOI_index=i;
    WOI=zeros(aoiX,aoiY);WOI(:,:)=missing; % initialize with NaN
    idx=find(WS==WOI_index); %find region of interest
    WOI(idx)=aoiSurfReady(idx); % fill in surface height values
    Z=reshape(WOI, [],1); %retains only the heights of the WOI, remaining entries are NaN

    % by computing surfnorm on whole surface, retains normal vectors for the
    % edge pixels
    nxoi=Nx(idx); nyoi=Ny(idx); nzoi=Nz(idx); % surface normal vectors for points in WOI
    Xtemp=X(~isnan(Z)); % Z contains heights of WOI
    Ytemp=Y(~isnan(Z));
    Ztemp=Z(~isnan(Z));

    mn=([nxoi,nyoi,nzoi]*Wind_dir)/(Wind_dir'*Wind_dir); % magnitude of component of the normal vector in the direction of the wind -- negative entries - gives components against the wind
    idx_mn=find(mn<0); % negative magnitude indicates opposite the wind
    %region3d=[Xtemp(idx_mn),Ytemp(idx_mn),Ztemp(idx_mn)]'; % pulls only points against the wind, restructure points to be (x location, y location, height) (in 3D)
    MN=([reshape(Nx,[],1),reshape(Ny,[],1), reshape(Nz,[],1)]*Wind_dir)/(Wind_dir'*Wind_dir);
    MN=reshape(MN,aoiSurfDims);
    MN_binary=double(MN<0);

    if isempty(Ztemp(idx_mn))
        WS_Height(i,1) = 0;
    else
        WS_Height(i,1)=max(Ztemp(idx_mn))-min(Ztemp(idx_mn));
    end

    % Take a unit square associated with each "against the wind" point and
    % compute the rotation matrix and projection matrix - take the
    % determinant of their composition to determine how the area will
    % change when it's normal is aligned with the surface normal and then
    % is projected on a plane perpendicular to the wind.
    trans_area=[];
    for j=1:length(idx_mn)
        k=idx_mn(j);
        ax=cross([0;0;1],[nxoi(k); nyoi(k); nzoi(k)]);  % find the axis of rotation by crossing the vertical normal (to a square) with the surface normal
        ax=ax/norm(ax);
        theta=acos([0;0;1]'*[nxoi(k); nyoi(k); nzoi(k)]); % find the angle of rotation
        % Find the rotation matrix using Rodrigues' formula
        K= [0, -ax(3), ax(2); ax(3), 0 -ax(1); -ax(2), ax(1), 0];
        Rot=eye(3)+sin(theta)*K+(1-cos(theta))*K*K; 
        A=orth(null(Wind_dir')); % find a basis for the plane orthogonal to the wind
        Proj=A*(A'*A)^(-1)*A'; % build the projection matrix onto the plane orthogonal to the wind
        % apply to generic unit square -- transform the vectors along two edges
        sq=[1,0;0,1; 0,0]*xyScale; % scale the unit square according to the scale of the data
        trans_pts=Proj*Rot*sq; % transform the square
        trans_area(j)=norm(cross(trans_pts(:,1),trans_pts(:,2))); % area of transformed square using the length of the cross product of two edges
    end
    WS_Sil_Area(i)=sum(trans_area);
    clear trans_area
end

WS_Area = regionprops(WS,"Area"); % list of areas of each watershed region
WS_Area = struct2cell(WS_Area);
WS_Area = cell2mat(WS_Area)';
WS_Area = WS_Area*xyScale^2; % scale converts from pixels to m^2

%--------------------------------------------------------------------------
%                                                Summarize WS measurements 
%--------------------------------------------------------------------------

Full_WS_Summary.area= WS_Area;
Full_WS_Summary.sil_area=WS_Sil_Area;
Full_WS_Summary.height= WS_Height;

%--------------------------------------------------------------------------
%                                                          Z_0 computation
%--------------------------------------------------------------------------

Z0 = .5*WS_Height.*WS_Sil_Area./WS_Area;
Full_WS_Summary.z0=Z0;

%--------------------------------------------------------------------------
%                                             Compute and Report Statistics
%--------------------------------------------------------------------------

Z0_Stats.mean = mean(Z0);
Z0_Stats.std = std(Z0);
% Compute skewness using method of moments to avoid Statistics Toolbox
% dependence
Z0_Stats.skew = sum((Z0 - Z0_Stats.mean) .^ 3)/double(num_WS) / Z0_Stats.std.^3;


fprintf('\n');
fprintf('\n***************************************************************')
fprintf('\n       Z0 Summary       ')
fprintf('\n  (Dataset : %s)', dataTitleText)
fprintf('\n***************************************************************')
fprintf('\n   number of watersheds : %i', num_WS)
fprintf('\n                Z0 mean : %5.3f (m)', Z0_Stats.mean)
fprintf('\n  Z0 standard deviation : %5.3f ', Z0_Stats.std)
fprintf('\n            Z0 skewness : %5.3f ', Z0_Stats.skew)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                           Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Plot watersheds colored by
% Z0 value


Z0_colored=zeros(fullX,fullY);
for i=1:num_WS
    idx=find(WS==i);
    Z0_colored(idx)= Z0(i);
end
cmax=max(max(max(Z0_colored))); % establish color
% scale with max value of max z0 when plotting z0
watershed_lines = NaN(aoiX,aoiY);
watershed_mask = WS==0;
%watershed_lines(watershed_mask==1)= min(min(aoiSurfReady)); % makes the
% watershed lines the lowest value on the surface so that they appear as a
% dark, visible line --- use for plotting with pcolor

watershed_lines(watershed_mask==1)= aoiSurfReady(watershed_mask==1); 

% figure;
% %t = tiledlayout(1, 3, "TileSpacing", "tight");
% subplot(1,2,1)
% imagesc(aoiSurfReady);
% hold on
% h=imagesc(watershed_lines); set(h, 'AlphaData',watershed_mask)
% axis square
% title('Data')
% colorbar()
% subplot(1,2,2)
% imagesc(Z0_colored); axis square
% caxis([0,cmax]);
% title('Watersheds Colored by Z_0 Value')
% colorbar();


figure;
surf(aoiSurfReady, 'EdgeColor', 'none', 'FaceColor', 'interp')
colormap(cmap)
hold on
surf(watershed_lines, 'EdgeColor', "black")
title([dataTitleText  ' with Watershed Boundaries'])

figure; colormap(cmap)
surf(aoiSurfReady,Z0_colored, 'EdgeColor','none')

title([dataTitleText  ' Colored by Z_0 Value'])
colorbar()

figure;
surf(aoiSurfReady,MN_binary)
colormap(wind_cmap)
title([dataTitleText ' Contrary to Wind Direction'])


% Plot a histogram of Z0 values
figure;
histogram(Z0, 10);
title([dataTitleText ' Z_0 Values'])


% Workspace cleanup - clear local working variables
clearvars -except Z0 Z0_Stats Full_WS_Summary num_WS num_WS aoiSurfReady cmap wind_cmap filterChoice dataFileName dataSurface dataTitleText1 xyScale WS Z0_colored MN_binary watershed_lines Wind_dir