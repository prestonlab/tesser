# Profile to be sourced before running tesser scripts.

case $USER in
    morton)
        STUDYDIR=$HOME/Dropbox/work/tesser
        export TESSER_BIDS=$HOME/Dropbox/work/tesser/bids
        export TESSER_FIGURES=$HOME/Dropbox/tesser_successor/Figures/v2
        export TESSER_RESULTS=$HOME/Dropbox/work/tesser/results
        conda activate tesser
        ;;

    mortonne)
        STUDYDIR=$STOCKYARD2/lonestar/tesser
        . $STOCKYARD2/lonestar/venv/tesser3/bin/activate
        export BATCHDIR=$STOCKYARD2/lonestar/tesser/batch/launchscripts
        ;;

    *)
        echo "Error: unknown user $USER."
        ;;
esac
export STUDYDIR
export SUBJNOS=100:101:102:103:104:105:106:107:108:109:110:111:112:113:114:115:116:117:119:120:121:122:123:124:125:126:127:128:129:130:131:132:133:135:136:137:138
export SUBJIDS=tesser_100:tesser_101:tesser_102:tesser_103:tesser_104:tesser_105:tesser_106:tesser_107:tesser_108:tesser_109:tesser_110:tesser_111:tesser_112:tesser_113:tesser_114:tesser_115:tesser_116:tesser_117:tesser_119:tesser_120:tesser_121:tesser_122:tesser_123:tesser_124:tesser_125:tesser_126:tesser_127:tesser_128:tesser_129:tesser_130:tesser_131:tesser_132:tesser_133:tesser_135:tesser_136:tesser_137:tesser_138
