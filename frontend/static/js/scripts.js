jQuery(document).ready(function() {
	
    /*
        Background slideshow
    */
    $('.top-content').backstretch("{{ url_for('static', filename='/img/1.jpg') }}");
    
    /*
        Wow
    */
    new WOW().init();
    
    /*
        Search form
    */
    $('.navbar-search-button .search-button').on('click', function(e){
    	e.preventDefault();
    	$(this).blur();
    	$('.navbar-menu-items, .navbar-search-form').toggleClass('disabled');
    	$('.navbar-search-form input.search').val('').focus();
    });
	
});