function add_description(node, content) {
    myFullDiagram.model.commit(function (m) { // this Model
        // This is the safe way to change model data
        // GoJS will be notified that the data has changed
        // and can update the node in the Diagram
        // and record the change in the UndoManager
        m.set(node.data, "description", content);

    }, "add description");
}

function update_nodes_for_study() {
    var ss1_key = ['4193', '5955', '4929', '4866'];
    var ss2_key = ['458', '751', '793', '575'];
    var ss3_key = ['21629', '24804', '22495', '23621'];


    var ss1 = ['This cluster performs image analysis - some of the tasks are image augmentation, running inference, getting image blob, preparing image for blob, detecting bbox in image, detecting bbox aspect ratio in image',
        'Some of the tasks of this cluster are creating input blobs for net, adding inference inputs, generate rpn on dataset with different range, initializing model from cfg',
        'Task of this cluster is running different variation of rcnn model like keypoint, mask, fast, generalized and also building generic detection model and building data parallel model. Also the models are run using GPU and Cuda library.',
        'Task of this cluster is extracting frozen features for fast rcnn, mask rcnn, fpn rpn. Building fast rcnn of RestNet50 and VGG16 and adding single parameter GPU update.'
    ];
    var ss2 = ['This cluster is preprocessing encoder, voxceleb2, speaker dirs, librispeech and converting wav to mel spectogram.',
        'This cluster is training synthesizer, tacotron, enqueing next train or test group, preparing token targets, batch',
        'This cluster is calculating similarity matrix, purge logs, inv linear spectogram, inv mel spectogram, drawing projections, saving wav, conversion of db to amp',
        'This cluster is preprocessing vocoder, running synthesis, getting hop size and calculating inv mel spectrogram, inv linear spectrogram'
    ];

    var ss3 = ['This cluster is preparing cookies, getting cookie header, merging environment settings and getting, sending requests.',
        'This cluster is building response, creating cookie, getting cookiejar from dict',
        'This cluster is handling 401 , preparing request, preparing body, preparing content length and building digest header.',
        'This cluster is deleting, sending, resolving redirects, merging cookies, getting full url.'
    ];


    // alert(subject_system);
    for (i = 0; i < 4; i++) {
        if (subject_system == 1) {
            var node = myFullDiagram.findNodeForKey(ss1_key[i]);
            // document.getElementById('cluster_description').innerHTML = ss1[i];
            // node.data.description = ss1[i];
            add_description(node, ss1[i]);
            // alert(ss1[i]);
        } else if (subject_system == 2) {
            var node = myFullDiagram.findNodeForKey(ss2_key[i]);
            // node.data.description = ss2[i];
            // document.getElementById('cluster_description').innerHTML = ss2[i];
            add_description(node, ss2[i]);
        } else if (subject_system == 3) {
            var node = myFullDiagram.findNodeForKey(ss3_key[i]);
            // node.data.description = ss3[i];
            // document.getElementById('cluster_description').innerHTML = ss3[i];
            add_description(node, ss3[i]);
        }

        var shape = node.findObject("SHAPE");
        shape.fill = "#87e8a1";

    }
    subject_system = subject_system + 1
}

function changeDoneClusterColor(key) {
    var node = myFullDiagram.findNodeForKey(key);
    var shape = node.findObject("SHAPE");
    shape.fill = "#ff66ff";
}

jQuery("#user_feedback").submit(function (event) {

    // Stop form from submitting normally
    event.preventDefault();

    // Get some values from elements on the page:
    var $form = $(this),
        t1 = $form.find("input[name='n_t1']:checked").val(),
        t2 = $form.find("input[name='n_t2']:checked").val(),
        t3 = $form.find("input[name='n_t3']:checked").val(),
        t4 = $form.find("input[name='n_t4']:checked").val(),
        t5 = $form.find("input[name='n_t5']:checked").val(),
        t6 = $form.find("input[name='n_t6']:checked").val(),
        f_key = $form.find("input[name='key']").val(),
        f_user_summary = $form.find("input[name='user_summary']").val(),
        f_comment = $form.find("input[name='comment']").val(),
        url = $form.attr("action");

    // alert(t1);
    // alert(t2);
    // alert(f_user_summary);
    // alert(f_comment);

    // Send the data using post
    var posting = $.post(url, {
        n_t1: t1,
        n_t2: t2,
        n_t3: t3,
        n_t4: t4,
        n_t5: t5,
        n_t6: t6,
        key: f_key,
        user_summary: f_user_summary,
        comment: f_comment
    });

    // Put the results in a div
    posting.done(function (data) {
        // var content = $( data ).find( "#content" );
        // $( "#result" ).empty().append( content );
        alert(data);
    });
    $("#user_feedback").trigger("reset");
    changeDoneClusterColor(f_key);
});

function showModal(part) {

    document.getElementById('modal-title').innerHTML = 'Cluster ID: ' + part.data.key;
    document.getElementById('tech1').innerHTML = part.data.tfidf_word;
    document.getElementById('tech2').innerHTML = part.data.tfidf_method;
    document.getElementById('tech3').innerHTML = part.data.lda_word;
    document.getElementById('tech4').innerHTML = part.data.lda_method;
    document.getElementById('tech5').innerHTML = part.data.lsi_word;
    document.getElementById('tech6').innerHTML = part.data.lsi_method;
    document.getElementById('key').value = part.data.key;
    document.getElementById('cluster_description').innerHTML = part.data.description;


    jQuery('#myModal').modal('show');
}